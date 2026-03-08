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
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import entropy
import time

# Audio and AI libraries
import librosa
import torch
from transformers import ClapModel, ClapProcessor

logging.basicConfig(level=logging.INFO)

DB_PATH = os.path.expanduser('~/.local/share/strawberry/strawberry/galaxy_embeddings.db')
MODEL_PATH = os.path.expanduser('~/.local/share/strawberry/strawberry/galaxy_umap_model.pkl')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "galaxy_config.json")

# Load Config
def load_config():
    with open(CONFIG_PATH, 'r') as f:
        data = json.load(f)
    return {c["label"]: tuple(c["pos"]) for c in data["audio"]}

GENRE_CENTROIDS = load_config()

# Prompts remain internal for CLAP model
VIBES_DATA = [
    {"label": "Electronic",      "prompt": "electronic dance music, synthesizers, drum machine, steady beats"},
    {"label": "DnB/Jungle",      "prompt": "drum and bass, jungle, breakbeats, fast electronic"},
    {"label": "Techno",          "prompt": "techno, four-on-the-floor, industrial beats, hypnotic"},
    {"label": "Synthwave/Retro", "prompt": "synthwave, retro 80s, analog synths, neon"},
    {"label": "Hardstyle",       "prompt": "hardstyle, distorted kick, jumping beats, aggressive dance"},
    {"label": "Rock",            "prompt": "rock music, electric guitar, acoustic drums, melodic vocals"},
    {"label": "Metal",           "prompt": "heavy metal, high-gain guitar, double-kick drums, intense"},
    {"label": "Punk/Garage",     "prompt": "punk rock, fast energy, electric chords, rebellious"},
    {"label": "Aggressive",      "prompt": "aggressive shouting, screaming vocals, distorted energy"},
    {"label": "Hip-Hop",         "prompt": "hip-hop, rap, rhythmic beats, vocal flow"},
    {"label": "Soul/R&B",        "prompt": "smooth soul, R&B, velvety vocals, groovy"},
    {"label": "Disco/Funk",      "prompt": "disco, funk, syncopated bass, dancey"},
    {"label": "Reggae/Dub",      "prompt": "reggae, off-beat guitar, deep rhythmic bass"},
    {"label": "Acoustic",        "prompt": "acoustic guitar, unplugged, warm natural vocals"},
    {"label": "Folk/Indie",      "prompt": "indie folk, storytelling, acoustic harmonies"},
    {"label": "Lo-Fi/Study",     "prompt": "lo-fi hip hop, vinyl crackle, relaxed atmosphere"},
    {"label": "Jazz/Lounge",     "prompt": "jazz music, brass harmonies, swing percussion"},
    {"label": "Blues",           "prompt": "blues, electric guitar bends, shuffle rhythm"},
    {"label": "Ambient",         "prompt": "ambient pads, ethereal soundscape, no beats"},
    {"label": "Classical",       "prompt": "classical orchestral, symphony, strings, woodwinds"},
    {"label": "Cinematic/Score", "prompt": "cinematic soundtrack, orchestral building, tension"},
    {"label": "Industrial",      "prompt": "industrial, metallic clanging, abrasive mechanical"},
    {"label": "Pop",             "prompt": "pop music, studio production, melodic hooks"},
    {"label": "Country",         "prompt": "country music, banjo, steel guitar, twangy vocals, honky tonk"},
    {"label": "Psytrance",       "prompt": "psytrance, goa, pulsating bassline, trippy synths, high energy"},
    {"label": "Shoegaze",        "prompt": "shoegaze, dream pop, ethereal vocals, walls of distorted guitar, reverb"},
    {"label": "Latin",           "prompt": "latin music, reggaeton, salsa, rhythmic percussion, spanish vocals"},
    {"label": "Glitch/IDM",      "prompt": "glitch, experimental electronic, digital stutter, clicks and pops"},
    {"label": "Vaporwave",       "prompt": "vaporwave, slowed down pop, nostalgic 80s aesthetic, muffled"}
]

VIBES_LIST = [v["prompt"] for v in VIBES_DATA]
VIBES_LABELS = [v["label"] for v in VIBES_DATA]
LYRICAL_LIST = ["song with vocals, singing, human voice, lyrics", "instrumental music, no vocals, no singing"]
ALL_PROMPTS = VIBES_LIST + LYRICAL_LIST

# Global AI references
clap_model = None
clap_processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
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

    reducer = umap.UMAP(n_components=2, n_neighbors=min(len(rows)-1, 30), min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(np.array(embeddings))
    x_min, x_max, y_min, y_max = np.min(coords[:,0]), np.max(coords[:,0]), np.min(coords[:,1]), np.max(coords[:,1])

    def scale(c, c_min, c_max):
        range_val = max(1e-6, c_max - c_min)
        return (c - (c_min + c_max) / 2.0) / range_val * 8500.0

    update_data = []
    for i in range(len(rows)):
        fx, fy = scale(coords[i, 0], x_min, x_max), scale(coords[i, 1], y_min, y_max)
        genre = v_genres[i]
        target = GENRE_CENTROIDS.get(genre)
        if target:
            pull = 0.85
            fx, fy = fx*(1-pull) + target[0]*pull, fy*(1-pull) + target[1]*pull
        update_data.append((float(fx), float(fy), paths[i]))

    conn.execute("BEGIN IMMEDIATE")
    conn.executemany("UPDATE embeddings SET x = ?, y = ? WHERE path = ?", update_data)
    conn.execute("COMMIT")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    
    # ... rest of main logic (processing audio)
    # The actual processing logic is omitted here for brevity but remains the same, 
    # just calling reduce_dimensions at the end.
    
    conn = setup_db(DB_PATH)
    reduce_dimensions(conn)
    conn.close()
