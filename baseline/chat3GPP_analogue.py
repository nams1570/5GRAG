import sys
sys.path.append("..")
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import chromadb
import json
from settings import config
#keyword search + BM 25 + cosine similarity between query & chunk embedding. RRF to ge weighted sum of BM25 and cosine score. 
# Retrn  top 1/10th o these chunks
# Rerank the 1/10th retrieved using  BGE-M3. Return top k2 of these reranked

#First retrieve top k1 with BM25

class Chat3GPPRetriever:
    def __init__(self,db_dir_path,collection_name,api_key=config["API_KEY"]):
        self.db_dir_path = db_dir_path
        self.api_key = api_key
        self.embeddings = OpenAIEmbeddings(
            model='text-embedding-3-large',
            api_key=self.api_key
        )
        self.collection_name = collection_name

        self.bm25Retriever = None
        self.all_documents = None

        self._setup_cosine_vector_store()
    
    def _setup_cosine_vector_store(self):
        self.chroma_client = chromadb.PersistentClient(path=self.db_dir_path)
        
        # Initialize vector store
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
    
    def _get_all_documents_in_db(self):
        if not self.vector_store:
            raise Exception("Vector store not initialized! Cannot get all documents")

        if self.all_documents:
            return self.all_documents
        collection = self.vector_store._collection
    
        results = collection.get(include=["documents", "metadatas"])
        documents = []
        for text, meta in zip(results.get("documents", []), results.get("metadatas", [])):
            documents.append(
                Document(
                    page_content=text,
                    metadata=meta if meta else {}
                )
            )
        self.all_documents = documents
        return self.all_documents
    
    def _build_bm25_retriever(self):
        bm25Retriever = BM25Retriever.from_documents(self._get_all_documents_in_db())
        return bm25Retriever
    
    def _compute_rrf_with_scores(self,bm25_results,cos_results,num_to_return):
        all_docs = {}
        cos_lookup = {}
        
        for rank,doc in enumerate(bm25_results):
            str_metadata = json.dumps(doc.metadata)
            dict_key = (doc.page_content,str_metadata)
            all_docs[dict_key] = {"bm25_rank": rank+1, "vector_rank": None}

        # Process Dense ranks
        for rank, (doc, cosine_score) in enumerate(cos_results):
            str_metadata = json.dumps(doc.metadata)
            dict_key = (doc.page_content,str_metadata)
            if dict_key not in all_docs:
                all_docs[dict_key] = {"bm25_rank":None, "vector_rank":rank+1}
            else:
                print("************\n\n\n\n")
                all_docs[dict_key]["vector_rank"] = rank+1
            cos_lookup[dict_key] = cosine_score
        
        for dict_key, ranks in all_docs.items():
            bm25_rank = ranks["bm25_rank"] if ranks["bm25_rank"] is not None else float("inf")
            vector_rank = ranks["vector_rank"] if ranks["vector_rank"] is not None else float("inf")
            
            rrf_score = (1 / (60 + bm25_rank)) + (1 / (60 + vector_rank))
            all_docs[dict_key]["rrf_score"] = rrf_score

        sorted_docs = sorted(all_docs.items(), key=lambda x: x[1]["rrf_score"], reverse=True)
        results = []
        for (doc_page_content,str_doc_metadata), fused_score in sorted_docs[:num_to_return]:
            doc_metadata = json.loads(str_doc_metadata)
            results.append({
                "doc": Document(page_content=doc_page_content,metadata=doc_metadata),
                "cosine_score": cos_lookup.get((doc_page_content,str_doc_metadata)),
                "fused_score": fused_score
            })
        return results
    
    def get_preranked_results(self,query,k1):
        if not self.bm25Retriever:
            self.bm25Retriever = self._build_bm25_retriever()
        #top k1 of bm25
        self.bm25Retriever.k = k1
        bm25_chunks =self.bm25Retriever.invoke(query)
        print(len(bm25_chunks))
        print("\n\n bm25 above^")
        cos_sim_chunks_with_Scores = self.vector_store.similarity_search_with_score(query,k=k1)
        
        preranked_results_from_rrf = self._compute_rrf_with_scores(bm25_results=bm25_chunks,cos_results=cos_sim_chunks_with_Scores,num_to_return=k1//10)
        return preranked_results_from_rrf


if __name__ == '__main__':
    c = Chat3GPPRetriever(db_dir_path='./db',collection_name='specs_and_discussions')
    query = "How many maximum 5QI we can create under one PDU Session?"
    print(c.get_preranked_results(query,20))

        
