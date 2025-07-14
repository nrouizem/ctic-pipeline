# purpose of this file is reducing memory usage through model injection

from functools import lru_cache
from sentence_transformers import SentenceTransformer, CrossEncoder

@lru_cache(maxsize=1)
def get_sentence_model():
    print("Loading sentence model...")
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

@lru_cache(maxsize=1)
def get_cross_encoder():
    print("Loading cross encoder...")
    return CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
