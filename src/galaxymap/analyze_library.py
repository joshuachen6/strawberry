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

# ─── EXPANDED VIBES DATA ───
VIBES_DATA = [
    {"label": "Electronic",      "centroid": (2000, 2000),   "prompt": "electronic dance music, synthesizers, drum machine, steady beats"},
    {"label": "DnB/Jungle",      "centroid": (2500, 1500),   "prompt": "drum and bass, jungle, breakbeats, fast electronic"},
    {"label": "Techno",          "centroid": (3000, 1000),   "prompt": "techno, four-on-the-floor, industrial beats, hypnotic"},
    {"label": "Synthwave/Retro", "centroid": (4000, 3000),   "prompt": "synthwave, retro 80s, analog synths, neon"},
    {"label": "Hardstyle",       "centroid": (3500, 500),    "prompt": "hardstyle, distorted kick, jumping beats, aggressive dance"},
    {"label": "Rock",            "centroid": (-2000, 2000),  "prompt": "rock music, electric guitar, acoustic drums, melodic vocals"},
    {"label": "Metal",           "centroid": (-3000, 3000),  "prompt": "heavy metal, high-gain guitar, double-kick drums, intense"},
    {"label": "Punk/Garage",     "centroid": (-2500, 1500),  "prompt": "punk rock, fast energy, electric chords, rebellious"},
    {"label": "Aggressive",      "centroid": (-4000, 4000),  "prompt": "aggressive shouting, screaming vocals, distorted energy"},
    {"label": "Hip-Hop",         "centroid": (2000, -2000),  "prompt": "hip-hop, rap, rhythmic beats, vocal flow"},
    {"label": "Soul/R&B",        "centroid": (1500, -2500),  "prompt": "smooth soul, R&B, velvety vocals, groovy"},
    {"label": "Disco/Funk",      "centroid": (2500, -1500),  "prompt": "disco, funk, syncopated bass, dancey"},
    {"label": "Reggae/Dub",      "centroid": (1000, -3000),  "prompt": "reggae, off-beat guitar, deep rhythmic bass"},
    {"label": "Acoustic",        "centroid": (-2000, -2000), "prompt": "acoustic guitar, unplugged, warm natural vocals"},
    {"label": "Folk/Indie",      "centroid": (-2500, -1500), "prompt": "indie folk, storytelling, acoustic harmonies"},
    {"label": "Lo-Fi/Study",     "centroid": (-1500, -2500), "prompt": "lo-fi hip hop, vinyl crackle, relaxed atmosphere"},
    {"label": "Jazz/Lounge",     "centroid": (-1000, -3000), "prompt": "jazz music, brass harmonies, swing percussion"},
    {"label": "Blues",           "centroid": (-800, -2000),  "prompt": "blues, electric guitar bends, shuffle rhythm"},
    {"label": "Ambient",         "centroid": (0, -3500),     "prompt": "ambient pads, ethereal soundscape, no beats"},
    {"label": "Classical",       "centroid": (0, 3500),      "prompt": "classical orchestral, symphony, strings, woodwinds"},
    {"label": "Cinematic/Score", "centroid": (1000, 4000),   "prompt": "cinematic soundtrack, orchestral building, tension"},
    {"label": "Industrial",      "centroid": (4000, -4000),  "prompt": "industrial, metallic clanging, abrasive mechanical"},
    {"label": "Pop",             "centroid": (0, 0),         "prompt": "pop music, studio production, melodic hooks"},
    {"label": "Country",         "centroid": (-3500, -1000), "prompt": "country music, banjo, steel guitar, twangy vocals, honky tonk"},
    {"label": "Psytrance",       "centroid": (4000, 1500),   "prompt": "psytrance, goa, pulsating bassline, trippy synths, high energy"},
    {"label": "Shoegaze",        "centroid": (-1500, 800),   "prompt": "shoegaze, dream pop, ethereal vocals, walls of distorted guitar, reverb"},
    {"label": "Latin",           "centroid": (1500, -2000),  "prompt": "latin music, reggaeton, salsa, rhythmic percussion, spanish vocals"},
    {"label": "Glitch/IDM",      "centroid": (3500, -3500),  "prompt": "glitch, experimental electronic, digital stutter, clicks and pops"},
    {"label": "Vaporwave",       "centroid": (3500, 3500),   "prompt": "vaporwave, slowed down pop, nostalgic 80s aesthetic, muffled"},

    # --- EXPANDED DISTRACTORS (The Sinks) ---
    {"label": "Distractor", "prompt": "background hiss, static noise, low quality recording"},
    {"label": "Distractor", "prompt": "person talking, speech, no music"},
    {"label": "Distractor", "prompt": "silence, faint ambient room tone"},
    {"label": "Distractor", "prompt": "sound effects, foley, nature sounds"},
    {"label": "Distractor", "prompt": "crowd noise, cheering, live applause"},
    {"label": "Distractor", "prompt": "wind blowing, rustling leaves, storm sounds"},
    {"label": "Distractor", "prompt": "engine idling, traffic noise, city sounds"},
    {"label": "Distractor", "prompt": "pouring rain, thunderstorm, water splashes"},
    {"label": "Distractor", "prompt": "metallic machinery, industrial clanking, non-musical rhythm"}
]

VIBES_LIST = [v["prompt"] for v in VIBES_DATA]
VIBES_LABELS = [v["label"] for v in VIBES_DATA]
GENRE_CENTROIDS = {v["label"]: v["centroid"] for v in VIBES_DATA if "centroid" in v}

LYRICAL_LIST = ["song with vocals, singing, human voice, lyrics", "instrumental music, no vocals, no singing"]
ALL_PROMPTS = VIBES_LIST + LYRICAL_LIST

# Global AI references
clap_model = None
clap_processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    # Increase timeout to 30s to handle contention with the main app
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None) # Use autocommit mode for manual control
    # Enable WAL mode for better concurrency
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    
    # Verify WAL mode
    res = conn.execute("PRAGMA journal_mode").fetchone()[0]
    if res.lower() != 'wal':
        logging.warning(f"Could not enable WAL mode, current mode: {res}")
        
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
            confidence_score REAL,
            lyrical_density REAL,
            temporal_movement REAL,
            harmonic_complexity REAL,
            genre_purity REAL,
            bpm_confidence REAL
        )
    ''')
    conn.commit()
    return conn

def download_model_if_needed():
    global clap_model, clap_processor
    if clap_model is None:
        logging.info(f"Loading CLAP model onto {device.upper()}...")
        # Use FP16 for massive speed boost on 4070
        clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device).half()
        clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

def prepare_audio_chunks(audio_path):
    """CPU Bound: Parallel audio loading across 32 threads."""
    try:
        try:
            full_duration = TinyTag.get(audio_path).duration
        except:
            full_duration = 180

        chunks = []
        midpoints = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        for pct in midpoints:
            try:
                chunk, _ = librosa.load(audio_path, sr=48000, offset=pct * full_duration, duration=7)
                if len(chunk) > 0:
                    peak = np.max(np.abs(chunk))
                    chunks.append(chunk * (0.89 / peak) if peak > 0 else chunk)
            except:
                continue
        return chunks if len(chunks) == 10 else None
    except Exception as e:
        logging.error(f"Failed to load {audio_path}: {e}")
        return None

def calculate_harmonic_complexity(chunks):
    if not chunks: return 0.0
    flatness_list = []
    chroma_vars = []
    for chunk in chunks:
        # chunk is at 48k
        f = librosa.feature.spectral_flatness(y=chunk)
        flatness_list.append(np.mean(f))
        c = librosa.feature.chroma_stft(y=chunk, sr=48000)
        chroma_vars.append(np.var(c))
    # Combine flatness (noise) and chroma var (complexity)
    return float((np.mean(flatness_list) + np.mean(chroma_vars)) / 2.0)

def extract_embedding_batch(batch_audio):
    """GPU Bound: High-throughput batch inference."""
    results = []
    if not batch_audio: return results

    # Flatten chunks for batching: [batch_size * 10 windows]
    flattened_chunks = [window for song in batch_audio if song for window in song]
    if not flattened_chunks: return [None] * len(batch_audio)

    # Move to GPU and cast to Half (FP16)
    inputs = clap_processor(text=ALL_PROMPTS, audio=flattened_chunks, return_tensors="pt", 
                            padding=True, sampling_rate=48000).to(device)
    if device == "cuda":
        inputs['input_features'] = inputs['input_features'].half()

    num_genres = len(VIBES_LIST)
    with torch.no_grad():
        outputs = clap_model(**inputs)
        # Reshape: [batch_size, 10_windows, num_prompts]
        logits = outputs.logits_per_audio.view(len(batch_audio), 10, -1)
        audio_embeds = outputs.audio_embeds.view(len(batch_audio), 10, -1)

        # 1. Genre Probabilities (Max-Pooling across windows)
        genre_logits = logits[:, :, :num_genres]
        max_genre_logits, _ = torch.max(genre_logits, dim=1)
        # Temperature 0.2 allows for more 'hybrid' overlap between similar genres
        probs_batch = (max_genre_logits / 0.2).softmax(dim=-1).cpu().numpy()
        
        # 2. Lyrical Density (Mean across windows for stability)
        lyrical_logits = logits[:, :, num_genres:]
        avg_lyrical_logits = torch.mean(lyrical_logits, dim=1)
        lyrical_probs = (avg_lyrical_logits / 0.2).softmax(dim=-1).cpu().numpy()
        lyrical_density_batch = lyrical_probs[:, 0] # Probability of "Vocal"

        # 3. Temporal Movement (StdDev of embeddings across windows)
        # We take the mean variation across the embedding dimensions
        temp_mov_batch = torch.std(audio_embeds, dim=1).mean(dim=-1).cpu().numpy()

        # 4. Average Audio Embeddings for spatial position
        avg_embeds_batch = torch.mean(audio_embeds, dim=1).cpu().numpy()

    for i in range(len(batch_audio)):
        probs = probs_batch[i]
        emb = avg_embeds_batch[i]
        conf = float(np.max(probs))
        label = VIBES_LABELS[np.argmax(probs)]
        
        # 5. Genre Purity (Entropy)
        # Lower entropy = higher purity
        purity = 1.0 - (entropy(probs) / np.log(len(probs)))
        
        # Classification Gate: Increase threshold slightly for purity
        genre = label if (label != "Distractor" and conf >= 0.15) else "Unclassified"
        # Weight 10.0 allows audio features (512-dim) to have more say relative to genre (25-dim)
        fused = np.concatenate([emb, probs * 10.0]).astype(np.float32)
        
        # Results: (fused, vibrancy, genre, conf, lyrical_density, temp_mov, purity)
        results.append({
            "fused": fused,
            "vibrancy": float(np.std(emb)),
            "genre": genre,
            "confidence": conf,
            "lyrical_density": float(lyrical_density_batch[i]),
            "temporal_movement": float(temp_mov_batch[i]),
            "genre_purity": float(purity)
        })
        
    return results

def retry_executemany(conn, sql, data, retries=10):
    for i in range(retries):
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany(sql, data)
            conn.execute("COMMIT")
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                logging.warning(f"Database locked, retrying ({i+1}/{retries})...")
                try: conn.execute("ROLLBACK")
                except: pass
                # Exponential backoff: starts at 1s, grows up to 10s per try
                time.sleep(min(10, i + 1))
                continue
            raise e
    raise sqlite3.OperationalError("Database remained locked after multiple retries")

def process_directory(directory, conn, force=False):
    cur = conn.cursor()
    cur.execute("SELECT path FROM embeddings")
    all_paths = [row[0] for row in cur.fetchall()]
    conn.execute("BEGIN IMMEDIATE")
    for path in all_paths:
        if not os.path.exists(path):
            cur.execute("DELETE FROM embeddings WHERE path = ?", (path,))
    conn.execute("COMMIT")
    
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
    
    # 4070 can handle batch_size of 16-24 easily
    batch_size = 16 
    with ThreadPoolExecutor(max_workers=32) as executor:
        for i in range(0, len(files_to_process), batch_size):
            batch = files_to_process[i : i + batch_size]
            
            # 1. CPU Stage: Parallel Audio Loading
            futures = [executor.submit(prepare_audio_chunks, f[0]) for f in batch]
            audio_data = [f.result() for f in futures]
            
            # 2. GPU Stage: Batch Inference
            results = extract_embedding_batch(audio_data)
            
            # 3. Librosa Stage: Harmonic Complexity (Parallel)
            harmonic_futures = [executor.submit(calculate_harmonic_complexity, audio) for audio in audio_data]
            harmonic_results = [f.result() for f in harmonic_futures]

            # 4. DB Stage: Sequential Persistence
            batch_updates = []
            for idx, res in enumerate(results):
                if res is None or res["fused"] is None: continue
                path, mtime = batch[idx]
                harm = harmonic_results[idx]
                batch_updates.append((
                    path, res["vibrancy"], datetime.datetime.now(), mtime, res["fused"].tobytes(), 
                    res["genre"], res["confidence"], res["lyrical_density"], res["temporal_movement"], 
                    harm, res["genre_purity"]
                ))
            
            if batch_updates:
                retry_executemany(conn, """
                    INSERT OR REPLACE INTO embeddings (
                        path, vibrancy, processed_at, mtime, raw_embedding, virtual_genre, confidence_score,
                        lyrical_density, temporal_movement, harmonic_complexity, genre_purity
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch_updates)
            prog = min(1.0, (i + batch_size) / len(files_to_process))
            print(json.dumps({"progress": prog, "status": f"Batched Analysis: {len(batch)} songs processed"}))
            sys.stdout.flush()
            
    return len(files_to_process)

def reduce_dimensions(conn):
    cur = conn.cursor()
    # Fetch everything in one go to avoid SELECT-in-loop (which increases lock time)
    cur.execute("SELECT path, raw_embedding, x, virtual_genre, genre_purity FROM embeddings WHERE raw_embedding IS NOT NULL")
    rows = cur.fetchall()
    if len(rows) < 2: return
        
    paths, embeddings, has_x, v_genres, purities = [], [], [], [], []
    for row in rows:
        paths.append(row[0])
        embeddings.append(np.frombuffer(row[1], dtype=np.float32))
        has_x.append(row[2] is not None)
        v_genres.append(row[3])
        purities.append(row[4] if row[4] is not None else 1.0)

    max_len = max(len(e) for e in embeddings)
    embeddings = np.array([np.pad(e, (0, max_len - len(e))) for e in embeddings])
    
    is_full_fit = True
    missing_indices = []

    if os.path.exists(MODEL_PATH):
        try:
            reducer, x_min, x_max, y_min, y_max = joblib.load(MODEL_PATH)
            missing_indices = [i for i, h in enumerate(has_x) if not h]
            # If less than 20% is missing, we do an incremental update
            if len(missing_indices) > 0 and len(missing_indices) <= 0.2 * len(rows):
                coords_missing = reducer.transform(embeddings[missing_indices])
                is_full_fit = False
        except Exception as e:
            logging.warning(f"Failed to load UMAP model, performing full fit: {e}")

    if is_full_fit:
        # High global structure (35 n_neighbors) + Low spreading (0.01 min_dist)
        reducer = umap.UMAP(
            n_components=2, 
            n_neighbors=35, 
            min_dist=0.01, 
            metric='cosine', 
            random_state=42,
            low_memory=True
        )
        coords_full = reducer.fit_transform(embeddings)
        x_min, x_max = np.min(coords_full[:,0]), np.max(coords_full[:,0])
        y_min, y_max = np.min(coords_full[:,1]), np.max(coords_full[:,1])
        joblib.dump((reducer, x_min, x_max, y_min, y_max), MODEL_PATH)
        final_coords = coords_full
    else:
        # Incremental: we don't change existing ones, just set the new ones
        # We need to build a full coords array where we only use the transform results for missing ones
        final_coords = np.zeros((len(rows), 2))
        # Note: We won't actually update existing ones, so we can just compute the new ones
        # but the normalization below needs to be consistent.
        # We'll calculate the normalized coords for just the missing ones.
        missing_coords_normalized = coords_missing # placeholder

    # Helper to scale coordinates to [-4000, 4000]
    def scale(c, c_min, c_max):
        if c_max <= c_min: return 0.0
        # ABSOLUTE MAPPING: Map the core 95% of the data to +/- 3000
        # This prevents outliers from squishing the main continents
        range_val = max(1e-6, c_max - c_min)
        return (c - c_min) / range_val * 6000.0 - 3000.0

    # Batch the updates to minimize database lock-wait cycles
    update_data = []
    indices_to_update = range(len(rows)) if is_full_fit else missing_indices
    
    for i in indices_to_update:
        path = paths[i]
        genre = v_genres[i]
        purity = purities[i]
        
        c_raw = final_coords[i] if is_full_fit else coords_missing[missing_indices.index(i)]
        fx = scale(c_raw[0], x_min, x_max)
        fy = scale(c_raw[1], y_min, y_max)
        
        if genre in GENRE_CENTROIDS and genre != "Unclassified":
            cx, cy = GENRE_CENTROIDS[genre]
            # Near-total anchoring for high purity (95% pull)
            # This ensures tracks cluster TIGHTLY around their labels
            pull = max(0.25, 0.95 * (purity ** 1.2)) 
            fx = fx * (1.0 - pull) + cx * pull
            fy = fy * (1.0 - pull) + cy * pull
            
        update_data.append((float(fx), float(fy), path))

    if update_data:
        retry_executemany(conn, "UPDATE embeddings SET x = ?, y = ? WHERE path = ?", update_data)

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
        print(json.dumps({"progress": 1.0, "status": "Projecting Galaxy (UMAP)..."}))
        sys.stdout.flush()
        reduce_dimensions(conn)
    conn.close()