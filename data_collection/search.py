from sentence_transformers import util
import json
import numpy as np

def search(query, model):
    q_emb = model.encode(query)
    embeddings = np.load("data_collection/data/embeddings.npy")
    scores = util.dot_score(q_emb, embeddings)[0].cpu().tolist()

    #Combine docs & scores
    with open("data_collection/data/records.json", 'r') as f:
        records = json.load(f)
    company_score_pairs = list(zip([record["company"] for record in records], scores))

    #Sort by decreasing score
    company_score_pairs = sorted(company_score_pairs, key=lambda x: x[1], reverse=True)
    return company_score_pairs