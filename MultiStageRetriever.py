from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain

class MultiStageRetriever:
    def __init__(self,llm,prompt_template):
        self.llm = llm
        self.doc_chain = create_stuff_documents_chain(self.llm, prompt_template)

    def reconstructDocChain(self,new_prompt_template):
        self.doc_chain = create_stuff_documents_chain(self.llm,new_prompt_template)

    def constructRetriever(self,db,selected_docs=None):
        """@selected_docs: list of documents to filter.
        @db: DBClient instance.
        @llm: the llm used for rewriting queries
        returns: Nothing"""
        if selected_docs is None or len(selected_docs) == 0:
            self.base_retriever = MultiQueryRetriever.from_llm(
                            retriever=db.getRetriever(), llm=self.llm
                        ) 
        else:
            # If we have selected one or more docs, then apply filtering
            name_list = [os.path.join(DOC_DIR,doc) for doc in selected_docs]
            print("name_list: ", name_list)
            name_filter = {"source": {"$in": name_list}}
            self.base_retriever = MultiQueryRetriever.from_llm(
                            retriever=db.getRetriever(search_kwargs={'filter': name_filter}), llm=self.llm
                        )
    def invoke(self,query,history):
        if not self.base_retriever:
            raise Exception("Error: No base retriever initialized. Has constructRetriever been run?")
        retrieved_docs = self.base_retriever.invoke(query)
        resp_answer = self.doc_chain.invoke({"context":retrieved_docs,"input":query,"history":history})
        resp = {"input":query,"history":history,"context":retrieved_docs,"answer":resp_answer}
        return resp
    
    