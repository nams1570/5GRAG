# Info
This subdirectory will contain the code for the baseline models we use to benchmark deepspecs.

# Chat3GPP_analogue
This is a best effort reconstruction of Chat3GPP, a RAG system for telecoms domain Q&A outlined in https://arxiv.org/abs/2501.13954, by Long Huang, Ming Zhao, Limin Xiao, Xiujun Zhang,  and Jungang Hu .

It uses both BM25 Retrieval and Cosine similarity search for preranking, combining their results using reciprocal rank fusion (rrf). The context is then reranked using BGE-M3 cross-encoder model.

We built this analogue rather than using their system directly because we wanted it to connect to our chromadb instance.


# simple_rag_controller
This is a simple RAG system set up for some testing.

