import threading
from sentence_transformers import util, SentenceTransformer, CrossEncoder
import json
import numpy as np
from rank_bm25 import BM25Okapi
from file_downloader import download_files_from_s3
from models import get_sentence_model, get_cross_encoder
import psutil, os
import gc

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
    _embeddings = np.load("data/embeddings_fp16.npy", mmap_mode="r")
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
           sem_top_k=2000,      # first stage semantic cut
           alpha=0.7,           # hybrid α weight
           top_k=1000,          # how many to return
           rerank_top_n=500):   # how many to rerank

    _ensure_initialized()

    def normalize(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn) if mx > mn else arr * 0

    # 1) Semantic stage: filter by type, then chunked top-k
    # flatten & filter indices_by_type
    all_type_idxs = [
        i
        for t in search_types
        for i in indices_by_type.get(t, [])
    ]
    if not all_type_idxs:
        return []

    # encode once as a tensor
    q_emb = model.encode(
        query + " " + " ".join(search_types),
        convert_to_tensor=True
    )

    # slice embeddings, run semantic_search
    emb_slice = _embeddings[all_type_idxs].astype(np.float32)    # mmap‑backed, cheap
    hits = util.semantic_search(
        query_embeddings=q_emb,
        corpus_embeddings=emb_slice,
        top_k=sem_top_k
    )[0]                                               # list of dicts

    # unpack hits → absolute record indices + normalized scores
    top_idxs      = [all_type_idxs[h['corpus_id']] for h in hits]
    top_sem_scores= np.array([h['score']        for h in hits])
    top_sem_norm  = normalize(top_sem_scores)

    # free big buffers
    del emb_slice, hits, top_sem_scores
    gc.collect()

    # 2) Lexical stage: BM25 only on top_idxs
    small_corpus = [records[i]["combined_text"].lower().split() for i in top_idxs]
    small_bm25        = BM25Okapi(small_corpus)

    lex_scores_small  = small_bm25.get_scores(query.lower().split())
    lex_norm_small    = normalize(lex_scores_small)

    # free corpus & bm25 if you like
    del small_corpus, small_bm25
    gc.collect()

    # 3) Combine & pick top_k
    combined = alpha * top_sem_norm + (1 - alpha) * lex_norm_small
    pick     = np.argsort(combined)[::-1][:top_k]
    candidates = [
        (records[top_idxs[i]], combined[i])
        for i in pick
    ]

    # 4) Optional Cross‑Encoder rerank on the head
    head    = candidates[:rerank_top_n]
    texts   = [r["combined_text"] for r,_ in head]
    queries = [query] * len(head)

    rerank_scores = _re_ranker.predict(list(zip(queries, texts)))
    reranked     = sorted(
        [(head[i][0], rerank_scores[i]) for i in range(len(head))],
        key=lambda x: x[1],
        reverse=True
    )

    final = reranked + candidates[rerank_top_n:]
    return final

def filter(company_score_pairs, doc_type):
    """
    Filter the keyword relevance data to return most relevant results.
    For now, returns top 50 with a threshold score.
    Could potentially return results that surpass some relevance threshold.
    """
    records = []
    for record, score in company_score_pairs:
        if len(records) == 50:
            break
        if score < -1:
            break
        if record["type"] == doc_type:
            records.append(record)
    return records