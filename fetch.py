import json

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