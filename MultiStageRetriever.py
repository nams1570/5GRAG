from importlib import metadata
from warnings import filters
from langchain.retrievers.multi_query import MultiQueryRetriever
from utils import RefObj,RetrieverResult, get_inclusive_tstmp_range, getDocIDFromText
from ReferenceExtractor import ReferenceExtractor
from HypotheticalDocGenerator import HypotheticalDocGenerator
import os 
from settings import config

RExt = ReferenceExtractor()
NUM_EXTRA_DOCS = config["NUM_EXTRA_DOCS"]
FILTER_START_TSTMP = '2000-01'

class MultiStageRetriever:
    def __init__(self,specDB,discussionDB=None,diffDB=None):
        self.selected_docs = None
        self.specDB = specDB
        self.discussionDB = discussionDB
        self.diffDB = diffDB

        self.hdg: HypotheticalDocGenerator = HypotheticalDocGenerator()

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
                if new_filter != {}:
                    filters.append(new_filter)
        if filters == []:
            return {}
        if len(filters) == 1:
            return filters[0]
        metadata_filter = {'$or':filters}
        return metadata_filter

    def getAdditionalContext(self,org_docs,db,hyp_doc):
        """@org_docs: list of initially retrieved document chunks from vector db.
        In this method, we parse the org_docs for external references and perform additional retrievals.
        returns: list of document chunks"""
        metadata_filter = self.buildFiltersFromRefs(docs=org_docs)
        if metadata_filter == {}:
            return []
        
        #metadataOnlyRetriever = db.getRetriever(search_kwargs={'filter':metadata_filter,'k':1000})
        additional_docs = []
        additional_docs.extend(db.vector_db.similarity_search(hyp_doc,NUM_EXTRA_DOCS,filter=metadata_filter))

        return additional_docs
    
    def retrieveFromSpecDB(self,hyp_doc:str):
        """We retrieve context info from the spec db.
        returns: first order and (possible) second order retrieval results"""
        org_docs = self.base_retriever.invoke(hyp_doc)
        print(f"There are {len(org_docs)}, and they are {org_docs}")
        if config["IS_SMART_RETRIEVAL"]:
            additional_docs = self.getAdditionalContext(org_docs,self.specDB,hyp_doc=hyp_doc)
            print(f"\n\n additional docs are {additional_docs}, and there are {len(additional_docs)} \n\n")
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
        docIDs = [diff.get("docID",None) for diff in diffs if diff.get("docID",None) is not None]
        return self.buildDocIDFilter(docIDs)
    
    def buildFiltersFromQuery(self,query):
        """This builds filters if there are docIDs in the query. 
        If there are no docIDs, returns empty dict."""
        docIDs = getDocIDFromText(query)
        return self.buildDocIDFilter(docIDs)
    
    def getFiltersForDiscussionDB(self,query:str)->dict:
        filters_from_query = self.buildFiltersFromQuery(query)
        if filters_from_query != {}:
            return filters_from_query
        
        diffRetriever = self.diffDB.getRetriever()
        diffs = diffRetriever.invoke(query)

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
        
        discussion_docs = self.discussionDB.vector_db.similarity_search(hyp_doc,config["NUM_REASONING_DOCS_TO_RETRIEVE"],filter=metadata_filter)
        if discussion_docs == []:
            discussion_docs = self.discussionDB.vector_db.similarity_search(hyp_doc,config["NUM_REASONING_DOCS_TO_RETRIEVE"])

        print(f"\n\n discussion_docs is \n**")
        print(discussion_docs)

        return discussion_docs

    def invoke(self,query):
        if not self.base_retriever:
            raise Exception("Error: No base retriever initialized. Has constructRetriever been run?")

        #invoke HypotheticalDocument here to get gpt response
        hyp_doc = self.hdg.generateHypotheticalDocument(query)
        if hyp_doc == None:
            hyp_doc = query

        org_docs,additional_docs = self.retrieveFromSpecDB(hyp_doc=hyp_doc)
        if self.discussionDB and (config["IS_SMART_RETRIEVAL"] == True):
            tdocs = self.retrieveReasoning(query,hyp_doc=hyp_doc)
        else:
            tdocs = []

        retriever_result = RetrieverResult(firstOrderSpecDocs=org_docs,secondOrderSpecDocs=additional_docs,retrievedDiscussionDocs=tdocs)
        return retriever_result
    
    