from sentence_transformers import util, SentenceTransformer, CrossEncoder
import json
import numpy as np
from rank_bm25 import BM25Okapi
from file_downloader import download_files_from_s3
from models import get_sentence_model, get_cross_encoder

download_files_from_s3()

with open("data/records.json", 'r') as f:
    records = json.load(f)

# Precompute a mapping from document‑type → list of indices in `records`
indices_by_type = {}
for idx, rec in enumerate(records):
    indices_by_type.setdefault(rec["type"], []).append(idx)

# prepare a tokenized corpus for BM25
_tokenized_corpus = [r["combined_text"].lower().split() for r in records]
_bm25 = BM25Okapi(_tokenized_corpus)

# load embeddings once
_embeddings = np.load("data/embeddings.npy", mmap_mode="r")

# Load a pretrained cross‑encoder for re‑ranking:
_re_ranker = get_cross_encoder()

def search(query, search_types, model,
           alpha=0.7,             # hybrid α weight
           top_k=500,             # how many to retrieve initially
           rerank_top_n=300):      # how many to re-rank

    # --- Stage 1: Semantic + Lexical over sliced subset -------------------

    # 1) encode the query
    q_emb = model.encode(query + " " + " ".join(search_types))

    # 2) figure out which record‑indices we actually care about
    #    (e.g. only "company" docs, or "deal" + "asset", etc.)
    mask_indices = sorted({
        i
        for doc_type in search_types
        for i in indices_by_type.get(doc_type, [])
    })
    if not mask_indices:
        return []

    # 3) slice out only those rows from the mmap'd embeddings
    emb_slice = _embeddings[mask_indices]

    # 4) compute semantic scores on the small slice
    sem_scores = util.dot_score(q_emb, emb_slice)[0].cpu().numpy()

    # 5) get full BM25 scores, then slice them too
    full_lex = _bm25.get_scores(query.lower().split())
    lex_scores = full_lex[mask_indices]

    # 6) normalize & combine
    def normalize(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn) if mx > mn else arr * 0

    sem_norm = normalize(sem_scores)
    lex_norm = normalize(lex_scores)
    combined = alpha * sem_norm + (1 - alpha) * lex_norm

    # 7) build the initial candidate list of (record, score)
    candidates = sorted(
        [(records[i], score) for i, score in zip(mask_indices, combined)],
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

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