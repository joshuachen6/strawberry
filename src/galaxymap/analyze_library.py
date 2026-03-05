import os
import sqlite3
import datetime
import argparse
import numpy as np
import umap
import json
from tinytag import TinyTag
from pathlib import Path
import logging
import joblib
import sys

# Audio processing
import librosa
import torch
from transformers import ClapModel, ClapProcessor

HAS_LIBROSA = True
HAS_CLAP = True

logging.basicConfig(level=logging.INFO)

DB_PATH = os.path.expanduser('~/.local/share/strawberry/strawberry/galaxy_embeddings.db')
MODEL_PATH = os.path.expanduser('~/.local/share/strawberry/strawberry/galaxy_umap_model.pkl')

# ─── REFINED VIBES DATA ───
VIBES_DATA = [
    {"label": "Electronic",      "centroid": (500, 500),   "prompt": "electronic dance music, synthesizers, drum machine, steady beats"},
    {"label": "DnB/Jungle",      "centroid": (600, 400),   "prompt": "drum and bass, jungle, breakbeats, fast electronic"},
    {"label": "Techno",          "centroid": (700, 300),   "prompt": "techno, four-on-the-floor, industrial beats, hypnotic"},
    {"label": "Synthwave/Retro", "centroid": (1000, 1000), "prompt": "synthwave, retro 80s, analog synths, neon"},
    {"label": "Hardstyle",       "centroid": (800, 200),   "prompt": "hardstyle, distorted kick, jumping beats, aggressive dance"},
    {"label": "Rock",            "centroid": (-500, 500),  "prompt": "rock music, electric guitar, acoustic drums, melodic vocals"},
    {"label": "Metal",           "centroid": (-700, 700),  "prompt": "heavy metal, high-gain guitar, double-kick drums, intense"},
    {"label": "Punk/Garage",     "centroid": (-600, 400),  "prompt": "punk rock, fast energy, electric chords, rebellious"},
    {"label": "Aggressive",      "centroid": (-800, 800),  "prompt": "aggressive shouting, screaming vocals, distorted energy"},
    {"label": "Hip-Hop",         "centroid": (500, -500),  "prompt": "hip-hop, rap, rhythmic beats, vocal flow"},
    {"label": "Soul/R&B",        "centroid": (400, -600),  "prompt": "smooth soul, R&B, velvety vocals, groovy"},
    {"label": "Disco/Funk",      "centroid": (600, -400),  "prompt": "disco, funk, syncopated bass, dancey"},
    {"label": "Reggae/Dub",      "centroid": (300, -700),  "prompt": "reggae, off-beat guitar, deep rhythmic bass"},
    {"label": "Acoustic",        "centroid": (-500, -500), "prompt": "acoustic guitar, unplugged, warm natural vocals"},
    {"label": "Folk/Indie",      "centroid": (-600, -400), "prompt": "indie folk, storytelling, acoustic harmonies"},
    {"label": "Lo-Fi/Study",     "centroid": (-400, -600), "prompt": "lo-fi hip hop, vinyl crackle, relaxed atmosphere"},
    {"label": "Jazz/Lounge",     "centroid": (-300, -700), "prompt": "jazz music, brass harmonies, swing percussion"},
    {"label": "Blues",           "centroid": (-200, -500), "prompt": "blues, electric guitar bends, shuffle rhythm"},
    {"label": "Ambient",         "centroid": (0, -800),    "prompt": "ambient pads, ethereal soundscape, no beats"},
    {"label": "Classical",       "centroid": (0, 800),     "prompt": "classical orchestral, symphony, strings, woodwinds"},
    {"label": "Cinematic/Score", "centroid": (200, 800),   "prompt": "cinematic soundtrack, orchestral building, tension"},
    {"label": "Industrial",      "centroid": (1000, -1000), "prompt": "industrial, metallic clanging, abrasive mechanical"},
    {"label": "Pop",             "centroid": (0, 0),       "prompt": "pop music, studio production, melodic hooks"},
    
    # Distractors (Sinks for non-music/noise)
    {"label": "Distractor", "prompt": "Low quality recording with heavy background hiss and static noise."},
    {"label": "Distractor", "prompt": "A person talking or speaking without any musical accompaniment."},
    {"label": "Distractor", "prompt": "Complete silence or very faint ambient room tone."},
    {"label": "Distractor", "prompt": "Non-musical sound effects, foley, or field recordings of nature."},
    {"label": "Distractor", "prompt": "Crowd noise, cheering, or applause from a live recording."}
]

VIBES_LIST = [v["prompt"] for v in VIBES_DATA]
VIBES_LABELS = [v["label"] for v in VIBES_DATA]

# FIX: Only collect centroids if they exist in the dictionary
GENRE_CENTROIDS = {v["label"]: v["centroid"] for v in VIBES_DATA if "centroid" in v}

clap_model = None
clap_processor = None

def setup_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            path TEXT PRIMARY KEY,
            song_id INTEGER,
            x REAL,
            y REAL,
            vibrancy REAL,
            processed_at TIMESTAMP,
            mtime REAL,
            raw_embedding BLOB,
            virtual_genre TEXT,
            confidence_score REAL
        )
    ''')
    conn.commit()
    return conn

def download_model_if_needed():
    global clap_model, clap_processor
    if clap_model is None:
        logging.info("Loading LAION-CLAP from Hugging Face...")
        clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

def extract_embedding(audio_path):
    embedding = None
    probs = np.zeros(len(VIBES_LIST))
    virtual_genre = "Unclassified"
    confidence_score = 0.0
    
    try:
        full_duration = TinyTag.get(audio_path).duration
    except:
        full_duration = 180

    all_logits = []
    all_audio_embeddings = []
    
    midpoints = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    for pct in midpoints:
        try:
            chunk, _ = librosa.load(audio_path, sr=48000, offset=pct * full_duration, duration=7)
        except:
            continue
        if len(chunk) == 0: continue
        
        peak = np.max(np.abs(chunk))
        chunk_norm = chunk * (0.89 / peak) if peak > 0 else chunk
        
        # FIX: Change 'audios' back to 'audio' as per deprecated API error
        inputs = clap_processor(text=VIBES_LIST, audio=chunk_norm, return_tensors="pt", padding=True, sampling_rate=48000)
        with torch.no_grad():
            outputs = clap_model(**inputs)
            all_logits.append(outputs.logits_per_audio)
            all_audio_embeddings.append(outputs.audio_embeds)
    
    if all_logits:
        stacked = torch.stack(all_logits)
        max_logits, _ = torch.max(stacked, dim=0) 
        
        sharpened = max_logits / 0.1
        probs = sharpened.softmax(dim=-1).numpy()[0]
        
        confidence_score = float(np.max(probs))
        top_idx = int(np.argmax(probs))
        winning_label = VIBES_LABELS[top_idx]
        
        if winning_label == "Distractor" or confidence_score < 0.12:
            virtual_genre = "Unclassified"
        else:
            virtual_genre = winning_label
    
    if all_audio_embeddings:
        avg_audio_emb = torch.mean(torch.stack(all_audio_embeddings), dim=0)
        embedding = avg_audio_emb.squeeze(0).numpy()

    if embedding is None:
        raise ValueError(f"Analysis failed for {audio_path}")

    vibrancy = float(np.std(embedding))
    weighted_probs = (probs * 15.0).astype(np.float32)
    fused_embedding = np.concatenate([embedding, weighted_probs])
    
    return fused_embedding, vibrancy, str(virtual_genre), confidence_score

def process_directory(directory, conn, force=False):
    cur = conn.cursor()
    cur.execute("SELECT path FROM embeddings")
    all_paths = [row[0] for row in cur.fetchall()]
    for path in all_paths:
        if not os.path.exists(path):
            cur.execute("DELETE FROM embeddings WHERE path = ?", (path,))
    conn.commit()
    
    files_to_process = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.mp3', '.flac', '.ogg', '.m4a', '.wav')):
                full_path = os.path.join(root, f)
                mtime = os.path.getmtime(full_path)
                cur.execute("SELECT mtime FROM embeddings WHERE path = ?", (full_path,))
                row = cur.fetchone()
                if force or not row or row[0] < mtime:
                    files_to_process.append((full_path, mtime))
                    
    if not files_to_process: return 0
        
    download_model_if_needed()
    for idx, (full_path, mtime) in enumerate(files_to_process):
        prog = (idx + 1) / len(files_to_process)
        print(json.dumps({"progress": prog, "status": f"Analyzing: {os.path.basename(full_path)}"}))
        sys.stdout.flush()
            
        emb, vibrancy, v_genre, conf = extract_embedding(full_path)
        if emb is not None:
            cur.execute("""
                INSERT OR REPLACE INTO embeddings (path, x, y, vibrancy, processed_at, mtime, raw_embedding, virtual_genre, confidence_score)
                VALUES (?, NULL, NULL, ?, ?, ?, ?, ?, ?)
            """, (full_path, vibrancy, datetime.datetime.now(), mtime, emb.tobytes(), v_genre, conf))
            conn.commit()
            
    return len(files_to_process)

def reduce_dimensions(conn):
    cur = conn.cursor()
    cur.execute("SELECT path, raw_embedding, x, virtual_genre FROM embeddings WHERE raw_embedding IS NOT NULL")
    rows = cur.fetchall()
    if len(rows) < 2: return
        
    paths, embeddings, has_x, v_genres = [], [], [], []
    for row in rows:
        paths.append(row[0])
        embeddings.append(np.frombuffer(row[1], dtype=np.float32))
        has_x.append(row[2] is not None)
        v_genres.append(row[3])

    max_len = max(len(e) for e in embeddings)
    embeddings = np.array([np.pad(e, (0, max_len - len(e))) for e in embeddings])
    
    needs_refit = True
    if os.path.exists(MODEL_PATH):
        try:
            reducer, x_min, x_max, y_min, y_max = joblib.load(MODEL_PATH)
            if sum(1 for h in has_x if not h) <= 0.2 * len(rows): needs_refit = False
        except: pass
            
    if needs_refit:
        reducer = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.001, metric='cosine', random_state=42)
        coords = reducer.fit_transform(embeddings)
        x_min, x_max = np.min(coords[:,0]), np.max(coords[:,0])
        y_min, y_max = np.min(coords[:,1]), np.max(coords[:,1])
        joblib.dump((reducer, x_min, x_max, y_min, y_max), MODEL_PATH)
    else:
        missing_indices = [i for i, h in enumerate(has_x) if not h]
        coords = reducer.transform(embeddings[missing_indices])
    
    xs = (coords[:,0] - x_min) / (x_max - x_min) * 2000.0 - 1000.0 if x_max > x_min else np.zeros(len(coords))
    ys = (coords[:,1] - y_min) / (y_max - y_min) * 2000.0 - 1000.0 if y_max > y_min else np.zeros(len(coords))

    for i, path in enumerate(paths if needs_refit else [paths[idx] for idx in missing_indices]):
        genre = v_genres[i] if needs_refit else v_genres[missing_indices[i]]
        fx, fy = xs[i], ys[i]
        
        if genre in GENRE_CENTROIDS and genre != "Unclassified":
            cx, cy = GENRE_CENTROIDS[genre]
            fx = fx * 0.7 + cx * 0.3
            fy = fy * 0.7 + cy * 0.3
            
        cur.execute("UPDATE embeddings SET x = ?, y = ? WHERE path = ?", (float(fx), float(fy), path))
    conn.commit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    
    conn = setup_db(DB_PATH)
    changed = process_directory(args.dir, conn, args.force)
    
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM embeddings WHERE x IS NULL")
    if changed > 0 or cur.fetchone()[0] > 0:
        print(json.dumps({"progress": 1.0, "status": "Projecting stars..."}))
        sys.stdout.flush()
        reduce_dimensions(conn)
    conn.close()