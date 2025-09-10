import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_core.documents import Document

"""Contains implementation of reranker that will be used in our chat3gpp analogue."""


def load_reranker(model_name="bge-reranker-large", device="cpu"):
    model_path = "BAAI/bge-reranker-large"
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map="auto").eval()
    return model, tokenizer

def get_rerank_scores(model, tokenizer, query, docs:list[Document]):
    pairs = [[query, doc.page_content] for doc in docs]
    
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(model.device)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

        scored_docs = list(zip(docs, scores.tolist()))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs