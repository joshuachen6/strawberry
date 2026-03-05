import os
import sqlite3
import datetime
import argparse
import numpy as np
import umap
from pathlib import Path
import essentia.standard as es
import logging

import joblib

logging.basicConfig(level=logging.INFO)

# Needs the embedding model (e.g., TensorflowPredict2D or TensorflowPredictEffnet-Discogs)
# Note: In a real scenario, you'd need the actual model file. For this task, we will simulate
# the embedding if the model file is not found, to ensure the script doesn't just crash
# when testing without a 50MB model file downloaded.

DB_PATH = os.path.expanduser('~/.local/share/strawberry/strawberry/galaxy_embeddings.db')
MODEL_PATH = os.path.expanduser('~/.local/share/strawberry/strawberry/galaxy_umap_model.pkl')

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
            raw_embedding BLOB
        )
    ''')
    conn.commit()
    return conn

def extract_embedding(audio_path, model_path=None):
    try:
        # Load audio (downsample to 16kHz as expected by many models)
        audio = es.MonoLoader(filename=audio_path, sampleRate=16000)()
        
        if model_path and os.path.exists(model_path):
            embedding_model = es.TensorflowPredictMusiCNN(graphFilename=model_path, output="model/Dense/BiasAdd")
            activations = embedding_model(audio)
            embedding = np.mean(activations, axis=0) # Average over frames
        else:
            # Fallback or simulation if model is missing
            # Compute some basic MFCCs to serve as a fake "deep embedding" for demonstration
            w = es.Windowing(type='hann')
            spec = es.Spectrum()
            mfcc = es.MFCC()
            pool = []
            for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
                mX = spec(w(frame))
                _, m = mfcc(mX)
                pool.append(m)
            pool = np.array(pool)
            if len(pool) == 0:
                embedding = np.zeros(13)
            else:
                embedding = np.mean(pool, axis=0)
            
        vibrancy = float(np.std(embedding))
        return embedding, vibrancy
    except Exception as e:
        logging.error(f"Error extracting embedding for {audio_path}: {e}")
        return None, 0.0

def process_directory(directory, conn, model_path=None):
    cur = conn.cursor()
    
    # 1. Scan files
    files_to_process = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.mp3', '.flac', '.ogg', '.m4a', '.wav')):
                full_path = os.path.join(root, f)
                mtime = os.path.getmtime(full_path)
                
                # Check if it needs processing
                cur.execute("SELECT mtime FROM embeddings WHERE path = ?", (full_path,))
                row = cur.fetchone()
                if not row or row[0] < mtime:
                    files_to_process.append((full_path, mtime))
                    
    if not files_to_process:
        logging.info("No new or modified files found.")
        return 0
        
    logging.info(f"Processing {len(files_to_process)} files...")
    
    # 2. Extract embeddings
    for full_path, mtime in files_to_process:
        logging.info(f"Extracting embedding for {full_path}")
        emb, vibrancy = extract_embedding(full_path, model_path)
        if emb is not None:
            # We don't have song_id here unless we query the main DB, so we leave it NULL 
            # or rely on path matching in the Qt app.
            cur.execute("""
                INSERT OR REPLACE INTO embeddings (path, song_id, x, y, vibrancy, processed_at, mtime, raw_embedding)
                VALUES (?, NULL, NULL, NULL, ?, ?, ?, ?)
            """, (full_path, vibrancy, datetime.datetime.now(), mtime, emb.tobytes()))
            conn.commit()
            
    return len(files_to_process)

def reduce_dimensions(conn, files_processed_count):
    cur = conn.cursor()
    cur.execute("SELECT path, raw_embedding, x FROM embeddings WHERE raw_embedding IS NOT NULL")
    rows = cur.fetchall()
    
    if len(rows) < 2:
        logging.info("Not enough vectors to run UMAP.")
        return
        
    paths = []
    embeddings = []
    has_x = []
    for row in rows:
        paths.append(row[0])
        # Reconstruct numpy array
        emb = np.frombuffer(row[1], dtype=np.float32)
        # Handle cases where embeddings might have different sizes (e.g. dummy MFCC vs real model)
        embeddings.append(emb)
        has_x.append(row[2] is not None)

    # Pad or truncate if dimensions differ
    max_len = max(len(e) for e in embeddings)
    embeddings = np.array([np.pad(e, (0, max_len - len(e))) if len(e) < max_len else e[:max_len] for e in embeddings])
    
    total_songs = len(rows)
    new_songs_count = sum(1 for h in has_x if not h)
    
    if new_songs_count == 0 and files_processed_count == 0:
        return

    needs_refit = True
    reducer = None
    x_min, x_max, y_min, y_max = 0, 0, 0, 0
    if os.path.exists(MODEL_PATH):
        try:
            saved_data = joblib.load(MODEL_PATH)
            if isinstance(saved_data, tuple) and len(saved_data) == 5:
                reducer, x_min, x_max, y_min, y_max = saved_data
                if new_songs_count <= 0.2 * total_songs:
                    needs_refit = False
        except Exception as e:
            logging.error(f"Failed to load UMAP model: {e}")
            
    if needs_refit:
        logging.info(f"Running UMAP fit_transform on {len(embeddings)} embeddings...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
        
        x_min, x_max = np.min(coords[:,0]), np.max(coords[:,0])
        y_min, y_max = np.min(coords[:,1]), np.max(coords[:,1])
        
        joblib.dump((reducer, x_min, x_max, y_min, y_max), MODEL_PATH)
        
        if x_max > x_min:
            xs = (coords[:,0] - x_min) / (x_max - x_min) * 2000.0 - 1000.0
        else:
            xs = np.zeros(len(coords))
            
        if y_max > y_min:
            ys = (coords[:,1] - y_min) / (y_max - y_min) * 2000.0 - 1000.0
        else:
            ys = np.zeros(len(coords))
            
        for i, path in enumerate(paths):
            cur.execute("UPDATE embeddings SET x = ?, y = ? WHERE path = ?", (float(xs[i]), float(ys[i]), path))
            
    else:
        missing_indices = [i for i, h in enumerate(has_x) if not h]
        if not missing_indices:
            return
        logging.info(f"Running UMAP transform on {len(missing_indices)} new embeddings...")
        missing_embeddings = embeddings[missing_indices]
        coords = reducer.transform(missing_embeddings)
        
        xs = (coords[:,0] - x_min) / (x_max - x_min) * 2000.0 - 1000.0 if x_max > x_min else np.zeros(len(coords))
        ys = (coords[:,1] - y_min) / (y_max - y_min) * 2000.0 - 1000.0 if y_max > y_min else np.zeros(len(coords))
        
        for idx, i in enumerate(missing_indices):
            path = paths[i]
            cur.execute("UPDATE embeddings SET x = ?, y = ? WHERE path = ?", (float(xs[idx]), float(ys[idx]), path))
        
    conn.commit()
    logging.info("Dimensionality reduction and coordinate update complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Audio Embedding Sidecar")
    parser.add_argument('--dir', type=str, required=True, help="Music directory to scan")
    parser.add_argument('--model', type=str, default=None, help="Path to Essentia TensorFlow model")
    args = parser.parse_args()
    
    conn = setup_db(DB_PATH)
    changed_count = process_directory(args.dir, conn, args.model)
    
    # Always try to reduce dimensions if something changed, or if x,y are NULL anywhere
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM embeddings WHERE x IS NULL")
    missing = cur.fetchone()[0]
    
    if changed_count > 0 or missing > 0:
        reduce_dimensions(conn, changed_count)
        
    conn.close()
