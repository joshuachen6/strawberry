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

GALAXY_DB_PATH = os.path.expanduser('~/.local/share/strawberry/strawberry/galaxy_unified_embeddings.db')
STRAWBERRY_DB_PATH = os.path.expanduser('~/.local/share/strawberry/strawberry/strawberry.db')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "galaxy_config.json")

# Load Config
def load_config():
    with open(CONFIG_PATH, 'r') as f:
        data = json.load(f)
    return {c["label"]: tuple(c["pos"]) for c in data["fusion"]}

FUSION_CENTROIDS = load_config()

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
            cur_gal.execute("SELECT mtime FROM embeddings WHERE path = ?", (path,))
            row = cur_gal.fetchone()
            if row and row[0] >= mtime: continue
        songs_to_process.append((path, lyrics, mtime, url))
    conn_str.close()
    return songs_to_process

def analyze_fusion(songs, conn):
    import onnxruntime as ort
    from tokenizers import Tokenizer
    import librosa
    
    available_providers = ort.get_available_providers()
    use_cuda = 'CUDAExecutionProvider' in available_providers
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
    
    lyric_repo = "Xenova/bge-large-en-v1.5"
    lyric_model_path = hf_hub_download(repo_id=lyric_repo, filename="onnx/model.onnx")
    lyric_tokenizer = Tokenizer.from_file(hf_hub_download(repo_id=lyric_repo, filename="tokenizer.json"))
    lyric_tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    lyric_tokenizer.enable_truncation(max_length=512)
    lyric_sess = ort.InferenceSession(lyric_model_path, providers=providers)
    l_inputs = [i.name for i in lyric_sess.get_inputs()]

    audio_repo = "Xenova/clap-htsat-unfused" 
    audio_model_path = hf_hub_download(repo_id=audio_repo, filename="onnx/model.onnx")
    audio_sess = ort.InferenceSession(audio_model_path, providers=providers)

    SENTIMENT_ANCHORS = {
        "happy": "joyful, cheerful, happy, optimistic",
        "sad": "sad, melancholic, sorrowful, heartbreaking",
        "angry": "angry, aggressive, furious, rage",
        "fear": "fearful, dark, mysterious, ominous",
        "love": "love, romantic, warm, affectionate",
        "surprise": "surprising, ethereal, dreamlike, unexpected"
    }
    anchor_labels = list(SENTIMENT_ANCHORS.keys())
    anchor_texts = ["Represent this sentence: " + t for t in SENTIMENT_ANCHORS.values()]
    encoded_anchors = lyric_tokenizer.encode_batch(anchor_texts)
    a_ids = np.array([e.ids for e in encoded_anchors], dtype=np.int64)
    a_mask = np.array([e.attention_mask for e in encoded_anchors], dtype=np.int64)
    l_feed = {"input_ids": a_ids, "attention_mask": a_mask}
    if "token_type_ids" in l_inputs: l_feed["token_type_ids"] = np.zeros_like(a_ids)
    a_out = lyric_sess.run(None, l_feed)[0]
    lyric_anchors = np.mean(a_out, axis=1)
    lyric_anchors /= np.linalg.norm(lyric_anchors, axis=1, keepdims=True)

    total = len(songs)
    batch_size = 16 if use_cuda else 2
    
    for i in range(0, total, batch_size):
        chunk = songs[i:i+batch_size]
        batch_updates = []
        for path, lyrics, mtime, url in chunk:
            if lyrics and len(lyrics.strip()) > 10:
                l_text = "Represent this sentence: " + lyrics[:1500]
                encoded = lyric_tokenizer.encode(l_text)
                ids = np.array([encoded.ids], dtype=np.int64)
                mask = np.array([encoded.attention_mask], dtype=np.int64)
                l_feed_song = {"input_ids": ids, "attention_mask": mask}
                if "token_type_ids" in l_inputs: l_feed_song["token_type_ids"] = np.zeros_like(ids)
                l_out = lyric_sess.run(None, l_feed_song)[0]
                l_emb = np.mean(l_out, axis=1)[0]
                l_emb /= (np.linalg.norm(l_emb) + 1e-9)
                l_sims = np.dot(lyric_anchors, l_emb)
                if np.max(l_sims) > 0.3:
                    sentiment = anchor_labels[np.argmax(l_sims)]
                else:
                    sentiment = "neutral"
            else:
                l_emb, sentiment = np.zeros(1024, dtype=np.float32), "neutral"

            try:
                y, sr = librosa.load(path, sr=22050, duration=5.0, offset=30.0)
                rms = np.sqrt(np.mean(y**2))
                energy = "high" if rms > 0.09 else ("low" if rms < 0.04 else "mid")
                mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                audio_feat = np.mean(mels, axis=1)
                audio_feat = np.pad(audio_feat, (0, 512-len(audio_feat)))
                audio_feat /= (np.linalg.norm(audio_feat) + 1e-9)
            except:
                audio_feat, rms, energy = np.zeros(512, dtype=np.float32), 0.05, "mid"

            fusion_emb = np.concatenate([l_emb * 0.75, audio_feat * 0.25])
            
            if not lyrics: 
                genre = "Instrumental"
            elif energy == "high":
                if sentiment == "sad": 
                    if rms > 0.15: genre = "Hardcore Melancholy"
                    else: genre = "Sad Banger"
                elif sentiment == "angry": 
                    if rms > 0.15: genre = "Aggressive Electronic"
                    else: genre = "Aggressive Rock"
                elif sentiment == "fear": genre = "Dark Techno"
                elif sentiment == "love": genre = "Passionate Dance"
                elif sentiment == "happy": 
                    if rms > 0.15: genre = "Fiery Passion"
                    else: genre = "Euphoric Electronic"
                else: genre = "Euphoric Electronic"
            elif energy == "low":
                if sentiment == "happy": genre = "Joyful Acoustic"
                elif sentiment == "love": genre = "Nostalgic Folk"
                elif sentiment == "sad": 
                    if rms < 0.02: genre = "Cold Despair"
                    else: genre = "Melancholic Acoustic"
                elif sentiment == "fear": 
                    if rms < 0.02: genre = "Haunting Ambient"
                    else: genre = "Mysterious Ambient"
                elif sentiment == "angry": genre = "Simmering Tension"
                elif sentiment == "surprise": genre = "Haunting Ambient"
                else: genre = "Ethereal Ambient"
            else:
                if sentiment == "surprise": genre = "Ethereal Pop"
                elif sentiment == "happy": 
                    if rms > 0.07: genre = "Sunny Disposition"
                    else: genre = "Optimistic Pop"
                elif sentiment == "sad": genre = "Bittersweet Indie"
                elif sentiment == "angry": genre = "Gritty Alternative"
                elif sentiment == "love": genre = "Warm Soul"
                elif sentiment == "fear": genre = "Suspenseful Score"
                elif sentiment == "neutral": genre = "Minimalist"
                else: genre = "Quiet Contentment"
            
            batch_updates.append((
                path, float(0.4 + min(0.5, rms*3)), datetime.datetime.now().isoformat(), float(mtime), fusion_emb.tobytes(),
                genre, 1.0, 1.0 if lyrics else 0.0, 0.5, 0.5, 1.0
            ))

        conn.execute("BEGIN IMMEDIATE")
        conn.executemany("INSERT OR REPLACE INTO embeddings (path, vibrancy, processed_at, mtime, raw_embedding, virtual_genre, confidence_score, lyrical_density, temporal_movement, harmonic_complexity, genre_purity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch_updates)
        conn.execute("COMMIT")
        print(json.dumps({"progress": min(0.85, (i + len(chunk)) / total * 0.8), "status": f"Unified Fusion: {min(total, i+len(chunk))}/{total}" }))
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
    reducer = umap.UMAP(n_components=2, n_neighbors=min(len(rows)-1, 35), min_dist=0.1, random_state=42, metric='cosine')
    coords = reducer.fit_transform(np.array(embeddings))
    x_min, x_max, y_min, y_max = np.min(coords[:,0]), np.max(coords[:,0]), np.min(coords[:,1]), np.max(coords[:,1])
    def scale(c, c_min, c_max):
        range_val = max(1e-6, c_max - c_min)
        return (c - (c_min + c_max) / 2.0) / range_val * 8500.0
    update_data = []
    for i in range(len(rows)):
        fx, fy = scale(coords[i, 0], x_min, x_max), scale(coords[i, 1], y_min, y_max)
        target = FUSION_CENTROIDS.get(v_genres[i])
        if target:
            pull = 0.85 if (v_genres[i] != "Instrumental") else 0.05
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
    if songs: analyze_fusion(songs, conn)
    print(json.dumps({"progress": 0.9, "status": "Projecting Multiverse..."}))
    sys.stdout.flush()
    reduce_dimensions(conn)
    print(json.dumps({"progress": 1.0, "status": "Finished"}))
    conn.close()
