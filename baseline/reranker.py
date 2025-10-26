import sys
sys.path.append("..")
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import Document
"""Contains implementation of reranker that will be used in our chat3gpp analogue."""
_reranker_model = None
_tokenizer = None

def load_reranker(model_name="bge-reranker-large", device="cpu"):
    global _reranker_model, _tokenizer
    if _reranker_model is not None and _tokenizer is not None:
        print(f"***********\n\n loading reranker model glob \n\n *********")
        return _reranker_model, _tokenizer
    model_path = "BAAI/bge-reranker-large"
    _tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    
    _reranker_model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map=device).eval()
    return _reranker_model, _tokenizer

def get_rerank_scores(model, tokenizer, query, docs:list[Document]):
    pairs = [[query, doc.page_content] for doc in docs]
    
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(model.device)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

        scored_docs = list(zip(docs, scores.tolist()))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs