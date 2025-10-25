from utils import RefObj,RetrieverResult, get_inclusive_tstmp_range, getDocIDFromText
from ReferenceExtractor import ReferenceExtractor
from HypotheticalDocGenerator import HypotheticalDocGenerator
from DBClient import DBClient
from settings import config

RExt = ReferenceExtractor()
NUM_EXTRA_DOCS = config["NUM_EXTRA_DOCS"]
FILTER_START_TSTMP = '2000-01'

class MultiStageRetriever:
    def __init__(self,pathToDB="../baseline/db",specCollectionName="specs_and_discussions",reasonCollectionName="reasoning_docs",diffCollectionName="diffs"):
        self.selected_docs = None

        self.hdg: HypotheticalDocGenerator = HypotheticalDocGenerator()

        self.collections: dict[str, DBClient] = {}
        self.collections["spec"] = DBClient(collection_name=specCollectionName,db_dir_path=pathToDB)
        self.collections["reasoning"] = DBClient(collection_name=reasonCollectionName,db_dir_path=pathToDB)
        self.collections["diff"] = DBClient(collection_name=diffCollectionName,db_dir_path=pathToDB)
        
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

    def getAdditionalContext(self,org_docs,hyp_doc,num_docs_to_retrieve):
        """@org_docs: list of initially retrieved document chunks from vector db.
        In this method, we parse the org_docs for external references and perform additional retrievals.
        returns: list of document chunks"""
        metadata_filter = self.buildFiltersFromRefs(docs=org_docs)
        if metadata_filter == {}:
            return []
        print(f"\n\n secondary retrieval filter is \n** {metadata_filter}")
        
        #metadataOnlyRetriever = db.getRetriever(search_kwargs={'filter':metadata_filter,'k':1000})
        additional_docs = []
        try:
            additional_docs.extend(self.collections["spec"].queryDB(query_text=hyp_doc,k=num_docs_to_retrieve,filter=metadata_filter))
        except Exception as e:
            print(f"error due to filter {metadata_filter}")
            raise e

        return additional_docs
    
    def retrieveFromSpecDB(self,hyp_doc:str):
        """We retrieve context info from the spec db.
        returns: first order and (possible) second order retrieval results"""
        org_docs = self.collections["spec"].queryDB(query_text=hyp_doc,k=config["NUM_DOCS_INITIAL_RETRIEVAL"])

        print(f"There are {len(org_docs)}, and they are {org_docs}")
        if config["IS_SMART_RETRIEVAL"] and config["NUM_EXTRA_DOCS"] > 0:
            secondary_retrieval_docs = []
            docs_to_build_filters_from = org_docs[:]
            #TODO: Revisit the recursion logic. what do we build filters from?
            #TODO: Revisit this logic. What to do if a call to additional context falls short of its budget?
            num_recursions = max(config["DEPTH"], 1)
            if NUM_EXTRA_DOCS < num_recursions:
                budget_per_recursion = 1
                num_recursions = NUM_EXTRA_DOCS
            else:
                budget_per_recursion = NUM_EXTRA_DOCS // num_recursions

            def _key(d):
                return (d.metadata.get("docID"), d.metadata.get("section"), d.page_content[:200])
            seen = set(_key(d) for d in org_docs)

            cumulative_extra_retrieval = (num_recursions * budget_per_recursion) - NUM_EXTRA_DOCS # If we do more retrievals than needed, we need to adjust
            for i in range(1,num_recursions+1,1):
                additional_docs = self.getAdditionalContext(docs_to_build_filters_from,hyp_doc=hyp_doc,num_docs_to_retrieve=budget_per_recursion)
                if len(additional_docs) < budget_per_recursion:
                    # If we didn't get enough docs, we need to adjust our budget
                    cumulative_extra_retrieval += (budget_per_recursion - len(additional_docs))

                # Dedupe additional docs
                for d in additional_docs:
                    k = _key(d)
                    if k not in seen:
                        seen.add(k)
                        secondary_retrieval_docs.append(d)

                docs_to_build_filters_from = additional_docs[:]

            print(f"\n\n secondary retrieval docs are {secondary_retrieval_docs}, and there are {len(secondary_retrieval_docs)} \n\n")
            desired_total = config["NUM_DOCS_INITIAL_RETRIEVAL"] + NUM_EXTRA_DOCS
            current_total = len(org_docs) + len(secondary_retrieval_docs)


            if desired_total > current_total:
                deficit = desired_total - current_total
                print("No additional docs could be retrieved based on references, getting some more based on similarity")
                org_docs = self.collections["spec"].queryDB(query_text=hyp_doc,k=(config["NUM_DOCS_INITIAL_RETRIEVAL"]+deficit))
        else:
           secondary_retrieval_docs = []

        return org_docs,secondary_retrieval_docs
    
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

        diffs = self.collections["diff"].queryDB(query_text=query,k=4)

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

        reasoning_docs = self.collections["reasoning"].queryDB(query_text=hyp_doc,k=config["NUM_REASONING_DOCS_TO_RETRIEVE"],filter=metadata_filter)
        if reasoning_docs == []:
            print("No reasoning docs could be retrieved based on diff/docID filtering, getting some more based on similarity")
            reasoning_docs = self.collections["reasoning"].queryDB(query_text=hyp_doc,k=config["NUM_REASONING_DOCS_TO_RETRIEVE"])

        print(f"\n\n reasoning_docs is \n**")
        print(reasoning_docs)

        return reasoning_docs

    def invoke(self,query):
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
    
    