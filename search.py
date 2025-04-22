import threading
from sentence_transformers import util, SentenceTransformer, CrossEncoder
import json
import numpy as np
from rank_bm25 import BM25Okapi
from file_downloader import download_files_from_s3
from models import get_sentence_model, get_cross_encoder
import psutil, os

process = psutil.Process(os.getpid())

def log_mem(step: str):
    """Print current RSS in MB with a label."""
    log = os.getenv("LOG_MEM")
    if not log:
        return
    rss = process.memory_info().rss / 1024**2
    print(f"[MEM] {step:30s}: {rss:7.1f} MB")

# ---- lazy initialization globals ----
_init_lock = threading.Lock()
_initialized = False

def _initialize_search_resources():
    global records, indices_by_type, _tokenized_corpus, _bm25, _embeddings, _re_ranker, _initialized

    log_mem("start")
    # 1) ensure we have the latest files
    download_files_from_s3()
    log_mem("after download_files_from_s3")

    # 2) load JSON
    with open("data/records.json", 'r') as f:
        records = json.load(f)
    log_mem("after loading JSON")

    # 3) build indices_by_type
    indices_by_type = {}
    for idx, rec in enumerate(records):
        indices_by_type.setdefault(rec["type"], []).append(idx)
    log_mem("after building indices_by_type")

    # 4) embeddings (memory‑map)
    _embeddings = np.load("data/embeddings.npy", mmap_mode="r")
    log_mem("after loading embeddings mmap")

    # 5) cross‑encoder
    _re_ranker = get_cross_encoder()
    log_mem("after get_cross_encoder()")

    _initialized = True
    log_mem("finished init")

def _ensure_initialized():
    with _init_lock:
        if not _initialized:
            _initialize_search_resources()

def search(query, search_types, model,
           sem_top_k=2000,         # first stage semantic cut
           alpha=0.7,             # hybrid α weight
           top_k=500,             # how many to retrieve initially
           rerank_top_n=300):      # how many to re-rank

    # lazy init
    _ensure_initialized()

    # --- Stage 1: Semantic + Lexical over sliced subset -------------------

    # normalize & combine
    def normalize(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn) if mx > mn else arr * 0

    # 1) encode query
    q_emb = model.encode(query + " " + " ".join(search_types))

    # 2) mask to your types
    mask_indices = sorted({
        i
        for doc_type in search_types
        for i in indices_by_type.get(doc_type, [])
    })
    if not mask_indices:
        return []

    # 3) embeddings slice + semantic scores
    emb_slice = _embeddings[mask_indices]
    sem_scores = util.dot_score(q_emb, emb_slice)[0].cpu().numpy()
    sem_norm = normalize(sem_scores)
    
    # 4) pick top semantically
    top_pos = np.argsort(sem_norm)[::-1][:sem_top_k]  # positions within `mask`
    top_indices = [mask_indices[i] for i in top_pos]
    top_sem_norm = sem_norm[top_pos]

    # build a small BM25 index
    small_corpus = [records[i]["combined_text"].lower().split() for i in top_indices]
    small_bm25   = BM25Okapi(small_corpus)
    lex_scores_small = small_bm25.get_scores(query.lower().split())
    lex_norm_small = normalize(lex_scores_small)

    combined = alpha * top_sem_norm + (1-alpha) * lex_norm_small
    # pick your final top_k
    pick = np.argsort(combined)[::-1][:top_k]
    candidates = [
        (records[top_indices[i]], combined[i])
        for i in pick
    ]
    log_mem("after stage 1")
    # --- Stage 2: Cross‑Encoder reranking on the head ----------------------

    # extract the top rerank_top_n candidates
    head = candidates[:rerank_top_n]
    texts = [r["combined_text"] for r, _ in head]
    queries = [query] * len(head)

    # batch predictions
    rerank_scores = _re_ranker.predict(list(zip(queries, texts)))

    # attach new scores & sort
    reranked = sorted(
        [(head[i][0], rerank_scores[i]) for i in range(len(head))],
        key=lambda x: x[1],
        reverse=True
    )

    # append the tail of the original (unstable) ranking
    tail = candidates[rerank_top_n:]
    final = reranked + tail
    log_mem("after stage 2")
    return final


def filter(company_score_pairs, doc_type):
    """
    Filter the keyword relevance data to return most relevant results.
    For now, returns top 20 or top n such that n.score > 0.
    Could potentially return results that surpass some relevance threshold.
    """
    records = []
    for record, score in company_score_pairs:
        if len(records) == 50:
            return records
        if record["type"] == doc_type:
            records.append(record)