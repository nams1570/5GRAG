import sys
sys.path.append("..")
from MultiStageRetriever import MultiStageRetriever
from ReferenceExtractor import ReferenceExtractor
from langchain_openai import OpenAIEmbeddings
from MetadataAwareChunker import getSectionedChunks,getFullSectionChunks,clean_file_name
from DBClient import DBClient
from settings import config
from utils import RefObj
from typing import Tuple
#get the retriever and client

RE = ReferenceExtractor()

def get_sections_from_docs(docs):
    sections = set()
    for doc in docs:
        sections.add(doc.metadata["section"])
    return list(sections)

def get_refs_without_tables(refs:list[RefObj])->list[RefObj]:
    final_refs = []
    for ref in refs:
        if "-" not in ref.reference:
            final_refs.append(ref)
    return final_refs

def get_chunks_with_refs(docs):
    """we exclude any table refs. 
    Returns only those chunks which have intra-doc clause references."""
    chunks = []
    chunk_to_refs = {}
    for chunk in docs:
        allRefs = get_refs_without_tables(RE.runREWithDocList(docs=[chunk]))
        section_names = RE.extractClauseNumbersOfSrc(allRefs)
        if section_names:
            chunk_to_refs[chunk.page_content] = section_names
            chunks.append(chunk)
    return chunks,chunk_to_refs

def get_all_existing_sections(ref_set:set,chunks)->Tuple[set,set]:
    all_refs_in_doc = set()
    for chunk in chunks:
        all_refs_in_doc.add(chunk.metadata["section"])
    return ref_set.intersection(all_refs_in_doc),all_refs_in_doc

def count_hit_rate_with_retrieval(chunks_in_file,org_chunk,db):
    """Calculates precision and recall, based on all of the `sections` retrieved by the retriever. 
    So tp,tn,fp,fn is calculated based on the number of sections not the number of chunks who meet criteria"""
    true_refs = get_refs_without_tables(RE.runREWithDocList([org_chunk]))
    true_refs = set(RE.extractClauseNumbersOfSrc(true_refs))
    print(f"true_refs before:{true_refs}")

    #ensure that only clauses that can be inthe document are accessed
    true_refs,all_sections_in_org_file = get_all_existing_sections(true_refs,chunks_in_file)
    print(f"true_refs:{true_refs}, all_refs are {all_sections_in_org_file}")
    org_docs,add_docs = mr.invoke(org_chunk.page_content,db)
    org_docs += add_docs
    #check the sections of the org_docs.  
    retriever_refs = get_refs_without_tables(RE.runREWithDocList(org_docs))
    retriever_refs = set(RE.extractClauseNumbersOfSrc(retriever_refs))
    print(f"retriever refs: {retriever_refs}")

    tp = len(retriever_refs.intersection(true_refs))
    fp = len(retriever_refs.difference(true_refs))
    fn =len(true_refs.difference(retriever_refs))
    
    true_complement = all_sections_in_org_file.difference(true_refs)
    retrieved_complement  = all_sections_in_org_file.difference(retriever_refs)
    tn = true_complement.intersection(retrieved_complement)

    precision = tp/(tp+fp) if tp+fp !=0 else 0.0
    recall = tp/(tp+fn) if tp+fn !=0 else 0.0
    f1_score = (2*precision*recall)/(precision+recall) if precision and recall else 0.0

    return {org_chunk.metadata["section"]:{"precision":precision,"recall":recall,"f1_score":f1_score}}

def get_avg_scores_for_file(file_name:str,results_dict:dict)->dict:
    tot_precision,tot_recall,tot_f1 = 0,0,0
    n = len(results_dict)
    for section in results_dict.keys():
        tot_precision += results_dict[section]["precision"]
        tot_recall += results_dict[section]["recall"]
        tot_f1 += results_dict[section]["f1_score"]
    return {"file_name":clean_file_name(file_name),"avg_precision":tot_precision/n,"avg_recall":tot_recall/n,"avg_f1":tot_f1/n}

if __name__ == "__main__":
    ## Setup classes
    mr = MultiStageRetriever()
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large',api_key=config["API_KEY"])
    db = DBClient(embedding_model=embeddings,db_dir_path="../pickles",doc_dir_path="../data")

    ## for each doc, get chunks and see hit rate
    file_list = ["../data/38211-i60.docx"]
    results = {}
    for file in file_list:
        mr.constructRetriever(db,selected_docs=[clean_file_name(file)])

        chunks = getFullSectionChunks([file])
        chunks_with_refs,_ = get_chunks_with_refs(chunks)
        for chunk_with_ref in chunks_with_refs:
            results = {**results,**count_hit_rate_with_retrieval(chunks,chunk_with_ref,db)}
        #print(results)
        print(get_avg_scores_for_file(file,results))
