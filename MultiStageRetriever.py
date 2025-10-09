from importlib import metadata
from warnings import filters
from langchain.retrievers.multi_query import MultiQueryRetriever
from utils import RefObj,RetrieverResult, get_inclusive_tstmp_range, getDocIDFromText
from ReferenceExtractor import ReferenceExtractor
from HypotheticalDocGenerator import HypotheticalDocGenerator
import os 
from settings import config
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_core.documents import Document

RExt = ReferenceExtractor()
NUM_EXTRA_DOCS = config["NUM_EXTRA_DOCS"]
FILTER_START_TSTMP = '2000-01'

class MultiStageRetriever:
    def __init__(self,pathToDB="../baseline/db",specCollectionName="specs_and_discussions",reasonCollectionName="reasoning_docs",diffCollectionName="diffs"):
        self.selected_docs = None

        self.hdg: HypotheticalDocGenerator = HypotheticalDocGenerator()
        self.chroma_client = chromadb.PersistentClient(path=pathToDB)

        embeddings = OpenAIEmbeddingFunction(model_name='text-embedding-3-large',api_key=config["API_KEY"]) #Since we're using openAI's llm, we have to use its embedding model

        
        # Initialize vector store
        self.collections = {}
        self.collections["spec"] = self.chroma_client.get_collection(name=specCollectionName, embedding_function=embeddings)
        self.collections["reasoning"] = self.chroma_client.get_collection(name=reasonCollectionName, embedding_function=embeddings)
        self.collections["diff"] = self.chroma_client.get_collection(name=diffCollectionName, embedding_function=embeddings)

    def constructRetriever(self,db,selected_docs=None):
        """@selected_docs: list of documents to filter.
        @db: DBClient instance.
        returns: Nothing"""
        if selected_docs is None or len(selected_docs) == 0:
            self.base_retriever = db.getRetriever(search_kwargs={"k":config["NUM_DOCS_INITIAL_RETRIEVAL"]})
            self.selected_docs = None
        else:
            # If we have selected one or more docs, then apply filtering
            name_list = selected_docs
            print("name_list: ", name_list)
            name_filter = {"source": {"$in": name_list}}
            self.selected_docs = name_list
            self.base_retriever = db.getRetriever(search_kwargs={"k":config["NUM_DOCS_INITIAL_RETRIEVAL"],'filter': name_filter})
        
    def queryDB(self,hyp_doc:str,k:int,filter:dict={},collectionType:str="spec")->list[Document]:
        """k is how many docs to retrieve, hyp_doc is what we query with"""
        if filter == {}:
            db_resp = self.collections[collectionType].query(query_texts=[hyp_doc],n_results=k)
        else:
            db_resp = self.collections[collectionType].query(query_texts=[hyp_doc],n_results=k,where=filter)

        docs = []
        for doc,meta in zip(db_resp['documents'][0],db_resp['metadatas'][0]):
            docs.append(Document(page_content=doc,metadata=meta))
        return docs

    def buildDocIdandSectionFilter(self,ref:RefObj,org_docid:str)->dict:
        section_names = RExt.extractClauseNumbersFromString(ref.reference)
        if ref.src == RExt.getSRCDOC():
            docId = org_docid
        else:
            docId = ref.src
        if section_names == []:
            return {}
        if docId == None:
            return {'section':{"$in":section_names}}
        filter = {'$and':[
                {'docID':{"$eq":docId}},
                {'section':{"$in":section_names}}
            ]}
        return filter

    def buildFiltersFromRefs(self,docs)->dict:
        """We want to make sure when we are resolving references, we search for chunks of the requisite section and same docid as  the reference."""
        filters = []
        for doc in docs:
            org_docid:str = doc.metadata.get("docID",None)
            ext_src:list[RefObj] = RExt.runREWithDocList(docs=[doc])
            for ref in ext_src:
                new_filter = self.buildDocIdandSectionFilter(ref,org_docid)
                if new_filter and new_filter not in filters:  # dedupe manually
                    filters.append(new_filter)
        if filters == []:
            return {}
        if len(filters) == 1:
            return filters[0]
        metadata_filter = {'$or':filters}
        return metadata_filter

    def getAdditionalContext(self,org_docs,hyp_doc):
        """@org_docs: list of initially retrieved document chunks from vector db.
        In this method, we parse the org_docs for external references and perform additional retrievals.
        returns: list of document chunks"""
        metadata_filter = self.buildFiltersFromRefs(docs=org_docs)
        if metadata_filter == {}:
            return []
        print(f"\n\n secondary retrieval filter is \n**")
        
        #metadataOnlyRetriever = db.getRetriever(search_kwargs={'filter':metadata_filter,'k':1000})
        additional_docs = []
        try:
            additional_docs.extend(self.queryDB(hyp_doc,NUM_EXTRA_DOCS,filter=metadata_filter,collectionType="spec"))
        except Exception as e:
            print(f"error due to filter {metadata_filter}")
            raise e

        return additional_docs
    
    def retrieveFromSpecDB(self,hyp_doc:str):
        """We retrieve context info from the spec db.
        returns: first order and (possible) second order retrieval results"""
        #org_docs = self.base_retriever.invoke(hyp_doc)
        org_docs = self.queryDB(hyp_doc=hyp_doc,k=config["NUM_DOCS_INITIAL_RETRIEVAL"],collectionType="spec")

        print(f"There are {len(org_docs)}, and they are {org_docs}")
        if config["IS_SMART_RETRIEVAL"] and config["NUM_EXTRA_DOCS"] > 0:
            additional_docs = self.getAdditionalContext(org_docs,hyp_doc=hyp_doc)
            print(f"\n\n additional docs are {additional_docs}, and there are {len(additional_docs)} \n\n")
            if len(additional_docs) == 0:
                print("No additional docs could be retrieved based on references, getting some more based on similarity")
                org_docs = self.queryDB(hyp_doc=hyp_doc,k=(config["NUM_DOCS_INITIAL_RETRIEVAL"]+NUM_EXTRA_DOCS),collectionType="spec")
        else:
            additional_docs = []

        return org_docs,additional_docs
    
    def buildTimestampFilter(self,toTimestamp):
        if toTimestamp == None:
            return {}
        
        validTimestampRange = get_inclusive_tstmp_range(FILTER_START_TSTMP,toTimestamp)
        timestampFilter = {'timestamp':{'$in':validTimestampRange}}
        return timestampFilter

    def buildDocIDFilter(self,docIDs:list[str])->dict:
        if docIDs == []:
            return {}
        if len(docIDs) == 1:
            docID = docIDs[0]
            filter = {'docID':{"$eq":docID}}
            return filter
        #more than one docID
        individualFilters = []
        for docID in docIDs:
            individualFilters.append({'docID':{"$eq":docID}})
        filter = {'$or':individualFilters}
        return filter
    
    def buildFiltersFromDiffs(self,diffs):
        docIDs = [diff.metadata.get("docID",None) for diff in diffs if diff.metadata.get("docID",None) is not None]
        return self.buildDocIDFilter(docIDs)
    
    def buildFiltersFromQuery(self,query):
        """This builds filters if there are docIDs in the query. 
        If there are no docIDs, returns empty dict."""
        docIDs = getDocIDFromText(query)
        print(f"\n\n docIDs from query \n **")
        print(docIDs)
        if docIDs == []:
            return {}
        return self.buildDocIDFilter(docIDs)
    
    def getFiltersForDiscussionDB(self,query:str)->dict:
        filters_from_query = self.buildFiltersFromQuery(query)
        print(f"\n\n filters from query \n **")
        print(filters_from_query)
        if filters_from_query != {}:
            return filters_from_query
        
        diffs = self.queryDB(hyp_doc=query,k=4,collectionType="diff")

        filters_from_diffs = self.buildFiltersFromDiffs(diffs)
        print(f"\n\ndiffs\n*****")
        print(diffs)

        if diffs == []:
            print("Nothing could be retrieved from diff db")
            return {}

        print(f"\n\n filters \n **")
        print(filters_from_diffs)
        return filters_from_diffs

    def retrieveReasoning(self,query,hyp_doc):
        """gets the change from diff db, and searches discussion db for relevant information on why the change was made.
        Returns: documents from discussion db"""
        #get diff similar to query
        metadata_filter = self.getFiltersForDiscussionDB(query)
        
        reasoning_docs = self.queryDB(hyp_doc,config["NUM_REASONING_DOCS_TO_RETRIEVE"],filter=metadata_filter,collectionType="reasoning")
        if reasoning_docs == []:
            print("No reasoning docs could be retrieved based on diff/docID filtering, getting some more based on similarity")
            reasoning_docs = self.queryDB(hyp_doc,config["NUM_REASONING_DOCS_TO_RETRIEVE"],collectionType="reasoning")

        print(f"\n\n reasoning_docs is \n**")
        print(reasoning_docs)

        return reasoning_docs

    def invoke(self,query):
        if not self.base_retriever:
            raise Exception("Error: No base retriever initialized. Has constructRetriever been run?")

        #invoke HypotheticalDocument here to get gpt response
        hyp_doc = self.hdg.generate_hypothetical_document(query)
        if hyp_doc == None:
            hyp_doc = query
        
        print(f"\n\n hypothetical doc is \n**")
        print(hyp_doc)

        org_docs,additional_docs = self.retrieveFromSpecDB(hyp_doc=hyp_doc)
        if config["NUM_REASONING_DOCS_TO_RETRIEVE"] != 0 and (config["IS_SMART_RETRIEVAL"] == True):
            tdocs = self.retrieveReasoning(query,hyp_doc=hyp_doc)
        else:
            tdocs = []

        retriever_result = RetrieverResult(firstOrderSpecDocs=org_docs,secondOrderSpecDocs=additional_docs,retrievedDiscussionDocs=tdocs)
        return retriever_result
    
    