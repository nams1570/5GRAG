import sys
sys.path.append("..")
from BM25Retriever import BM25Retriever
from utils import Document
from RAGQAEngine import RAGQAEngine
import time
from DBClient import DBClient
import json
from settings import config
from baseline.reranker import get_rerank_scores, load_reranker
#keyword search + BM 25 + cosine similarity between query & chunk embedding. RRF to ge weighted sum of BM25 and cosine score. 
# Retrn  top 1/10th o these chunks
# Rerank the 1/10th retrieved using  BGE-M3. Return top k2 of these reranked

#First retrieve top k1 with BM25

class Chat3GPPRetriever:
    def __init__(self,db_dir_path,collection_name,api_key=config["API_KEY"]):
        self.db_dir_path = db_dir_path
        self.api_key = api_key
        self.collection_name = collection_name

        self.bm25Retriever = BM25Retriever()
        self.all_documents = None
        
        self.dbclient = DBClient(embedding_model_name='text-embedding-3-large',collection_name=collection_name,db_dir_path=db_dir_path)

        self.reranker_model = None
        self.tokenizer = None

        print(f"***********\n\n init: building bm25 retriever \n\n *********")
        self._build_bm25_retriever()
        print(f"***********\n\n loading reranker model \n\n *********")
        self.reranker_model, self.tokenizer = load_reranker()
    
    def _get_all_documents_in_db(self):
        if not self.dbclient:
            raise Exception("Vector store not initialized! Cannot get all documents")

        if self.all_documents:
            return self.all_documents
    
        documents = self.dbclient.getAllDocsFromDB()
        self.all_documents = documents
        return self.all_documents
    
    def _build_bm25_retriever(self):
        if not self.bm25Retriever:
            raise Exception("BM25 Retriever not initialized! Cannot build BM25 retriever")
        self.bm25Retriever.from_documents(self._get_all_documents_in_db())
    
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

    def _compute_cosine_sim_scores(self,query,k1):
        cos_sim_chunks_with_Scores = self.dbclient.queryDBWithScores(query_text=query,k=k1)
        results = []
        for doc,meta,dist in zip(cos_sim_chunks_with_Scores['documents'][0],cos_sim_chunks_with_Scores['metadatas'][0],cos_sim_chunks_with_Scores['distances'][0]):
            results.append((Document(page_content=doc,metadata=meta), 1 - dist)) #cosine similarity = 1 - cosine distance
        return results

    
    def get_preranked_results(self,query,k1):
        #top k1 of bm25
        bm25_chunks =self.bm25Retriever.invoke(query,k=k1)
        print(len(bm25_chunks))
        print("\n\n bm25 above^")

        cos_sim_chunks_with_Scores = self._compute_cosine_sim_scores(query,k1)
        print(f"cos scores \n \n {len(cos_sim_chunks_with_Scores)}")

        preranked_results_from_rrf = self._compute_rrf_with_scores(bm25_results=bm25_chunks,cos_results=cos_sim_chunks_with_Scores,num_to_return=k1//10)
        return preranked_results_from_rrf

    def rerank(self,preranked_results, query, k2):
        #must pass langchain Documents as docs to get_rerank_scores
        reranked_results = get_rerank_scores(model=self.reranker_model,tokenizer=self.tokenizer,query=query,docs=preranked_results)
        return reranked_results[:k2]

    def invoke(self,query,k1,k2):
        """@k1; the number of docs retrieved by BM25 and cosine similarity each in preranking. 
        @k2: the number of reranked documents returned."""
        preranked_results = self.get_preranked_results(query,k1)
        preranked_results = [result["doc"] for result in preranked_results]

        reranked_results = self.rerank(preranked_results,query=query,k2=k2)
        reranked_results = [result[0] for result in reranked_results]
        return reranked_results

class Chat3GPPAnalogue:
    def __init__(self,db_dir_path,collection_name,api_key=config["API_KEY"], model_name=config["MODEL_NAME"]):
        self.retriever =  Chat3GPPRetriever(db_dir_path=db_dir_path,collection_name=collection_name,api_key=api_key)
        
        self.qa_engine = RAGQAEngine(prompt_template_file_path="../prompt.txt",model_name=model_name,api_key=api_key)
    
    def runController(self,question,k1,k2):
        retrieved_docs = self.retriever.invoke(question,k1=k1,k2=k2)
        answer = self.qa_engine.get_answer_from_context(question,retrieved_docs)
        
        return answer, retrieved_docs
    
    def getOnlyRetrievalResults(self,question,k1,k2):
        retrieved_docs = self.retriever.invoke(question,k1=k1,k2=k2)
        answer = ""
        return answer, retrieved_docs


if __name__ == '__main__':
    c = Chat3GPPAnalogue(db_dir_path='./db',collection_name='specs_and_discussions')
    query = "How many maximum 5QI we can create under one PDU Session?"
    start = time.time()
    print(c.runController(query,100,5))
    end = time.time()
    print(f"\n\nTime taken {end-start} seconds\n\n")
    query = "How many maximum 5QI we can create under two PDU Sessions?"
    start = time.time()
    print(c.runController(query,100,5))
    end = time.time()
    print(f"\n\nTime taken {end-start} seconds\n\n")

        
