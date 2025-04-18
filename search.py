from sentence_transformers import util, SentenceTransformer, CrossEncoder
import json
import numpy as np
from rank_bm25 import BM25Okapi

with open("data/records.json", 'r') as f:
    records = json.load(f)

# prepare a tokenized corpus for BM25
_tokenized_corpus = [r["combined_text"].lower().split() for r in records]
_bm25 = BM25Okapi(_tokenized_corpus)

# load embeddings once
_embeddings = np.load("data/embeddings.npy")

# Load a pretrained cross‑encoder for re‑ranking:
_re_ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

def search(query, search_types, model,
           alpha=0.7,             # hybrid α weight
           top_k=100,             # how many to retrieve initially
           rerank_top_n=50):      # how many to re-rank

    # --- Stage 1: Hybrid semantic + lexical ---
    q_emb = model.encode(query + " " + " ".join(search_types))
    sem_scores = util.dot_score(q_emb, _embeddings)[0].cpu().numpy()
    lex_scores = _bm25.get_scores(query.lower().split())
    
    # normalize & combine (as before) …
    def normalize(x):
        min_x, max_x = x.min(), x.max()
        return (x - min_x) / (max_x - min_x) if max_x > min_x else x
    sem_norm = normalize(sem_scores)
    lex_norm = normalize(lex_scores)
    combined_scores = alpha * sem_norm + (1 - alpha) * lex_norm

    # pair & sort, then take top_k
    candidates = sorted(zip(records, combined_scores),
                        key=lambda x: x[1], reverse=True)[:top_k]

    # --- Stage 2: Cross‑Encoder re‑ranking on the head of the list ---
    # Prepare input pairs for re‑ranking
    rerank_candidates = candidates[:rerank_top_n]
    texts = [r["combined_text"] for r, _ in rerank_candidates]
    queries = [query] * len(texts)

    # Batch‐predict relevance scores
    rerank_scores = _re_ranker.predict(list(zip(queries, texts)))

    # Attach those scores and re‐sort
    for idx, ((record, _), score) in enumerate(zip(rerank_candidates, rerank_scores)):
        rerank_candidates[idx] = (record, score)

    # Final ranking: top reranked first, then the remainder of stage‑1
    reranked = sorted(rerank_candidates, key=lambda x: x[1], reverse=True)
    remainder = candidates[rerank_top_n:]
    final_results = reranked + remainder

    return final_results


def filter(company_score_pairs, doc_type):
    """
    Filter the keyword relevance data to return most relevant results.
    For now, returns top 20 or top n such that n.score > 0.
    Could potentially return results that surpass some relevance threshold.
    """
    records = []
    for record, score in company_score_pairs:
        if len(records) == 20 or score < 0:
            return records
        if record["type"] == doc_type:
            records.append(record)