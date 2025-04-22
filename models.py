# purpose of this file is reducing memory usage through model injection

from functools import lru_cache
from sentence_transformers import SentenceTransformer, CrossEncoder

@lru_cache(maxsize=1)
def get_sentence_model():
    print("SENTENCE MODEL")
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

@lru_cache(maxsize=1)
def get_cross_encoder():
    print("CROSS ENCODER")
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
