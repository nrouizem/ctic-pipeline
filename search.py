from sentence_transformers import util, SentenceTransformer
import json
import numpy as np

def search(query):
    model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
    q_emb = model.encode(query)
    embeddings = np.load("data_collection/data/embeddings.npy")
    scores = util.dot_score(q_emb, embeddings)[0].cpu().tolist()

    #Combine docs & scores
    with open("data_collection/data/records.json", 'r') as f:
        records = json.load(f)
    company_score_pairs = list(zip(records, scores))

    #Sort by decreasing score
    company_score_pairs = sorted(company_score_pairs, key=lambda x: x[1], reverse=True)
    return company_score_pairs

def filter(company_score_pairs, doc_type):
    """
    Filter the keyword relevance data to return most relevant results.
    For now, just return the top 10 results.
    Could potentially return results that surpass some relevance threshold.
    """
    return [record["company"] for record, score in company_score_pairs[:10] if record["type"] == doc_type]