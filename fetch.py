import requests
from bs4 import BeautifulSoup
import pandas as pd
import glob
import json
from concurrent.futures import ThreadPoolExecutor
from duckduckgo_search import DDGS
import time
from functools import singledispatch
import matplotlib.pyplot as plt
import csv
from tqdm import trange
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup
import requests
from sec_api import FullTextSearchApi
from datetime import datetime
import os


def bpd_relevance(keywords, crawl=True):
    """
    Returns keyword relevance.
    """
    if crawl:
        with open("data/BPD/crawled_data.json", "r") as f:
            data = json.load(f)
    else:
        with open("data/BPD/not_crawled_data.json", "r") as f:
            data = json.load(f)
    result = {}
    for company_dict in data:
        total_occurrences = 0
        wordcount = 0
        for page in company_dict["pages"]:
            total_occurrences += sum(page["text"].count(keyword) for keyword in keywords)
            wordcount += len(page["text"].split())
        score = total_occurrences / (wordcount + 1)
        result[f"{company_dict["company_name"]}"] = score
    
    result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    return list(result.keys())

def filter(results):
    """
    Filter the keyword relevance data to return most relevant results.
    For now, just return the top 5 results.
    Could potentially return results that surpass some relevance threshold.
    """
    return results[:5]