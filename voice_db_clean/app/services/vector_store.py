import numpy as np
import faiss
import json
import os

DIM = 192
INDEX_PATH = "data/faiss.index"
META_PATH = "data/meta.json"

index = faiss.IndexFlatIP(DIM)
names = []

def identify_speaker(embedding):
    if index.ntotal == 0:
        return None, 0.0

    emb = embedding / np.linalg.norm(embedding)
    scores, ids = index.search(emb.reshape(1, -1), 1)

    best_score = float(scores[0][0])
    best_id = int(ids[0][0])

    if best_id == -1:
        return None, 0.0

    return names[best_id], best_score

def load_store():
    global index, names
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r") as f:
            names[:] = json.load(f)

def save_store():
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w") as f:
        json.dump(names, f)

def add_embedding(embedding, person_name):
    emb = embedding / np.linalg.norm(embedding)
    index.add(emb.reshape(1, -1))
    names.append(person_name)
    save_store()

def verify_speaker(embedding, person_name):
    emb = embedding / np.linalg.norm(embedding)

    scores = []

    for i, name in enumerate(names):
        if name == person_name:
            stored_emb = index.reconstruct(i)
            score = np.dot(emb, stored_emb)
            scores.append(score)

    if not scores:
        return 0.0

    return float(max(scores))
