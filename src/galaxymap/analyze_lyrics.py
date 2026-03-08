import os
import sqlite3
import datetime
import argparse
import numpy as np
import umap
import json
import logging
import sys
import time
import urllib.parse
from urllib.parse import urlparse
from huggingface_hub import hf_hub_download

# AI / GPU optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def fix_nvidia_paths():
    venv_base = os.path.dirname(os.path.dirname(sys.executable))
    found_paths = []
    for lib_dir in ["lib", "lib64"]:
        nvidia_base = os.path.join(venv_base, lib_dir, f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia")
        if os.path.exists(nvidia_base):
            for root, dirs, files in os.walk(nvidia_base):
                if "lib" in dirs:
                    found_paths.append(os.path.join(root, "lib"))
    if found_paths:
        lp = os.environ.get("LD_LIBRARY_PATH", "")
        system_paths = [p for p in lp.split(":") if not ("nvidia" in p.lower() and "compat" in p.lower())]
        os.environ["LD_LIBRARY_PATH"] = ":".join(found_paths + system_paths)

fix_nvidia_paths()
logging.basicConfig(level=logging.INFO)

GALAXY_DB_PATH = os.path.expanduser('~/.local/share/strawberry/strawberry/galaxy_lyrics_embeddings.db')
STRAWBERRY_DB_PATH = os.path.expanduser('~/.local/share/strawberry/strawberry/strawberry.db')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "galaxy_config.json")

# Load Config
def load_config():
    with open(CONFIG_PATH, 'r') as f:
        data = json.load(f)
    return {c["label"]: tuple(c["pos"]) for c in data["lyrics"]}

EMOTION_CENTROIDS = load_config()

def setup_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            path TEXT PRIMARY KEY, song_id INTEGER, x REAL, y REAL, vibrancy REAL,
            processed_at TIMESTAMP, mtime REAL, raw_embedding BLOB, virtual_genre TEXT,
            confidence_score REAL, lyrical_density REAL, temporal_movement REAL,
            harmonic_complexity REAL, genre_purity REAL, bpm_confidence REAL
        )
    ''')
    conn.commit()
    return conn

def get_songs_from_strawberry(force=False, conn_galaxy=None):
    if not os.path.exists(STRAWBERRY_DB_PATH): return []
    conn_str = sqlite3.connect(STRAWBERRY_DB_PATH)
    cur_str = conn_str.cursor()
    cur_str.execute("SELECT url, lyrics, mtime FROM songs")
    all_rows = cur_str.fetchall()
    songs_to_process = []
    cur_gal = conn_galaxy.cursor()
    for url, lyrics, mtime in all_rows:
        if not url: continue
        path = urllib.parse.unquote(urlparse(url).path)
        if not path: continue
        if not force:
            cur_gal.execute("SELECT mtime, virtual_genre FROM embeddings WHERE path = ?", (path,))
            row = cur_gal.fetchone()
            if row and row[0] >= mtime and (row[1] != "Instrumental" or not lyrics):
                continue
        songs_to_process.append((path, lyrics, mtime))
    conn_str.close()
    return songs_to_process

def analyze_lyrics_onnx_heavy(songs, conn):
    import onnxruntime as ort
    from tokenizers import Tokenizer
    
    # Check GPU via ONNX Runtime available providers
    available_providers = ort.get_available_providers()
    use_cuda = 'CUDAExecutionProvider' in available_providers
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
    
    model_repo = "Xenova/bge-large-en-v1.5"
    model_path = hf_hub_download(repo_id=model_repo, filename="onnx/model.onnx")
    tokenizer_path = hf_hub_download(repo_id=model_repo, filename="tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=512)
    
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        device_name = "GPU (CUDA)" if 'CUDAExecutionProvider' in session.get_providers() else "CPU"
    except Exception as e:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        device_name = "CPU"

    l_inputs = [i.name for i in session.get_inputs()]

    ANCHOR_DESCRIPTIONS = {
        "Joyful": "A song full of absolute happiness, sunshine, celebration, energy, and pure euphoric joy.",
        "Love/Warm": "A song about deep romantic love, intimacy, caring, warmth, and tender affection.",
        "Surprise": "An ethereal, mysterious, magical, and dreamlike experience. Something surreal and unexpected.",
        "Sadness": "A deeply melancholic, lonely, heartbreaking, and sorrowful song filled with grief and pain.",
        "Anger": "An aggressive, furious, violent, and hateful song filled with rage and intense conflict.",
        "Fear/Dark": "A terrifying, dark, spooky, and ominous song filled with dread, shadows, and anxiety.",
        "Neutral": "A regular, everyday, objective, and calm song with no strong emotional bias."
    }
    anchor_labels = list(ANCHOR_DESCRIPTIONS.keys())
    anchor_texts = ["Represent this sentence for searching relevant passages: " + t for t in ANCHOR_DESCRIPTIONS.values()]
    encoded_anchors = tokenizer.encode_batch(anchor_texts)
    a_ids = np.array([e.ids for e in encoded_anchors], dtype=np.int64)
    a_mask = np.array([e.attention_mask for e in encoded_anchors], dtype=np.int64)
    l_feed = {"input_ids": a_ids, "attention_mask": a_mask}
    if "token_type_ids" in l_inputs: l_feed["token_type_ids"] = np.zeros_like(a_ids)
    a_out = session.run(None, l_feed)[0]
    anchor_embs = np.mean(a_out, axis=1)
    anchor_embs = anchor_embs / np.linalg.norm(anchor_embs, axis=1, keepdims=True)

    total = len(songs)
    batch_size = 16 if device_name != "CPU" else 4
    for i in range(0, total, batch_size):
        chunk = songs[i:i+batch_size]
        texts = ["Represent this sentence for searching relevant passages: " + s[1][:1500] if s[1] else "" for s in chunk]
        valid_indices = [idx for idx, text in enumerate(texts) if len(text) > 60]
        valid_texts = [texts[idx] for idx in valid_indices]
        chunk_labels = ["Neutral"] * len(texts)
        chunk_confs = [0.0] * len(texts)
        chunk_embeddings = [np.zeros(1024, dtype=np.float32)] * len(texts)
        if valid_texts:
            encoded = tokenizer.encode_batch(valid_texts)
            ids = np.array([encoded.ids], dtype=np.int64)
            mask = np.array([encoded.attention_mask], dtype=np.int64)
            l_feed_song = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in l_inputs: l_feed_song["token_type_ids"] = np.zeros_like(ids)
            out = session.run(None, l_feed_song)[0]
            embs = np.mean(out, axis=1)
            embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
            for v_idx, emb in zip(valid_indices, embs):
                chunk_embeddings[v_idx] = emb
                sims = np.dot(anchor_embs, emb)
                top_idx = np.argmax(sims)
                if top_idx == 6 and np.max(sims[:6]) > 0.2:
                    top_idx = np.argmax(sims[:6])
                chunk_labels[v_idx] = anchor_labels[top_idx]
                chunk_confs[v_idx] = float(sims[top_idx])
        batch_updates = []
        for idx, (path, lyrics, mtime) in enumerate(chunk):
            genre = chunk_labels[idx] if lyrics else "Instrumental"
            conf = chunk_confs[idx] if lyrics else 1.0
            vibrancy = max(0.35, conf * 1.4) if lyrics else 0.15
            batch_updates.append((
                path, float(vibrancy), datetime.datetime.now().isoformat(), float(mtime), chunk_embeddings[idx].tobytes(),
                genre, float(conf), 1.0 if lyrics else 0.0, 0.5, 0.5, 1.0
            ))
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany("INSERT OR REPLACE INTO embeddings (path, vibrancy, processed_at, mtime, raw_embedding, virtual_genre, confidence_score, lyrical_density, temporal_movement, harmonic_complexity, genre_purity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch_updates)
        conn.execute("COMMIT")
        print(json.dumps({"progress": min(0.85, (i + len(chunk)) / total * 0.8), "status": f"AI Deep Scan: {min(total, i+len(chunk))}/{total} ({device_name})" }))
        sys.stdout.flush()

def reduce_dimensions(conn):
    cur = conn.cursor()
    cur.execute("SELECT path, raw_embedding, virtual_genre FROM embeddings WHERE raw_embedding IS NOT NULL")
    rows = cur.fetchall()
    if len(rows) < 3: return
    paths, embeddings, v_genres = [], [], []
    for row in rows:
        paths.append(row[0])
        embeddings.append(np.frombuffer(row[1], dtype=np.float32))
        v_genres.append(row[2])
    reducer = umap.UMAP(n_components=2, n_neighbors=min(len(rows)-1, 40), min_dist=0.1, random_state=42, metric='cosine')
    coords = reducer.fit_transform(np.array(embeddings))
    x_min, x_max, y_min, y_max = np.min(coords[:,0]), np.max(coords[:,0]), np.min(coords[:,1]), np.max(coords[:,1])
    def scale(c, c_min, c_max):
        range_val = max(1e-6, c_max - c_min)
        return (c - (c_min + c_max) / 2.0) / range_val * 8500.0
    update_data = []
    for i in range(len(rows)):
        fx, fy = scale(coords[i, 0], x_min, x_max), scale(coords[i, 1], y_min, y_max)
        target = EMOTION_CENTROIDS.get(v_genres[i])
        if target:
            pull = 0.85 if (v_genres[i] != "Neutral" and v_genres[i] != "Instrumental") else 0.15
            fx, fy = fx*(1-pull) + target[0]*pull, fy*(1-pull) + target[1]*pull
        update_data.append((float(fx), float(fy), paths[i]))
    conn.execute("BEGIN IMMEDIATE")
    conn.executemany("UPDATE embeddings SET x = ?, y = ? WHERE path = ?", update_data)
    conn.execute("COMMIT")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    conn = setup_db(GALAXY_DB_PATH)
    songs = get_songs_from_strawberry(args.force, conn)
    if songs: analyze_lyrics_onnx_heavy(songs, conn)
    print(json.dumps({"progress": 0.9, "status": "Projecting Semantic Galaxy..."}))
    sys.stdout.flush()
    reduce_dimensions(conn)
    print(json.dumps({"progress": 1.0, "status": "Finished"}))
    conn.close()
