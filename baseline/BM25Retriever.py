from rank_bm25 import BM25Okapi

def basic_text_splitter(text):
    return text.split()

class BM25Retriever:
    def __init__(self):
        self.preprocess_func = basic_text_splitter

    def from_documents(self, documents):
        print(f"bm25 retriever from_documents called with {len(documents)} documents")
        self.docs = documents
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)

        texts = [self.preprocess_func(text) for text in texts
                 ]
        self.vectorizer = BM25Okapi(texts)

    def invoke(self,query,k):
        processed_query = self.preprocess_func(query)
        results = self.vectorizer.get_top_n(processed_query,self.docs,n=k)
        return results