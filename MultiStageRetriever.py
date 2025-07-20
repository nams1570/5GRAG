from langchain.retrievers.multi_query import MultiQueryRetriever
from utils import RefObj
from ReferenceExtractor import ReferenceExtractor
import os 
from settings import config

RExt = ReferenceExtractor()
NUM_EXTRA_DOCS = config["NUM_EXTRA_DOCS"]

class MultiStageRetriever:
    def __init__(self):
        self.selected_docs = None

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

    def buildDocIdandSectionFilter(ref:RefObj,org_docid):
        section_names = RExt.extractClauseNumbersFromString(ref.reference)
        if ref.src == RExt.getSRCDOC():
            docId = org_docid
        else:
            docId = ref.src
            
        if docId == None:
            return {'section':{"$in":section_names}}
        filter = {'$and':[
                {'docID':{"$eq":docId}},
                {'section':{"$in":section_names}}
            ]}
        return filter

    def buildFiltersFromRefs(self,docs)->list[RefObj]:
        """We want to make sure when we are resolving references, we search for chunks of the requisite section and same docid as  the reference."""
        ext_src: list[RefObj] = []
        for doc in docs:
            org_docid = doc.metadata.get("docID",None)
            ext_src.extend(RExt.runREWithDocList(docs=[doc]))
        return ext_src

    def getAdditionalContext(self,org_docs,db):
        """@org_docs: list of initially retrieved document chunks from vector db.
        In this method, we parse the org_docs for external references and perform additional retrievals.
        returns: list of document chunks"""
        ext_src: list[RefObj] = self.buildFiltersFromRefs(docs=org_docs)
        section_names = []
        section_names = RExt.extractClauseNumbersOfSrc(ext_src)
        print(f"\n extr_src is {ext_src}\n")
        print(f"section_names are {section_names}")
        if section_names == []:
            return []

        if self.selected_docs:
            metadata_filter = {'$and':[
                {'source':{'$in':self.selected_docs}},
                {'section':{"$in":section_names}}
            ]}
        else:    
            metadata_filter = {"section":{"$in":section_names}}
        
        #metadataOnlyRetriever = db.getRetriever(search_kwargs={'filter':metadata_filter,'k':1000})
        additional_docs = []
        additional_docs.extend(db.vector_db.similarity_search("",NUM_EXTRA_DOCS,filter=metadata_filter))

        return additional_docs

    def invoke(self,query,db):
        if not self.base_retriever:
            raise Exception("Error: No base retriever initialized. Has constructRetriever been run?")
        org_docs = self.base_retriever.invoke(query)
        print(f"There are {len(org_docs)}, and they are {org_docs}")
        if config["IS_SMART_RETRIEVAL"]:
            additional_docs = self.getAdditionalContext(org_docs,db)
            print(f"\n\n additional docs are {additional_docs}, and there are {len(additional_docs)} \n\n")
        else:
            additional_docs = []

        return org_docs,additional_docs
    
    