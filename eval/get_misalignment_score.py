import sys

sys.path.append("..")
from MultiStageRetriever import MultiStageRetriever
from ReferenceExtractor import ReferenceExtractor
from langchain_openai import OpenAIEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import json
from tqdm import tqdm
from MetadataAwareChunker import getSectionedChunks,getFullSectionChunks,clean_file_name
from DBClient import DBClient
from settings import config
from utils import RefObj, RetrieverResult
from typing import Tuple
from CollectionNames import SPECS_AND_DISCUSSIONS as SPEC_COLL_NAME,REASONING_DOCS as TDOC_COLL_NAME,DIFFS as DIFF_COLL_NAME

#get the retriever and client

RE = ReferenceExtractor()
DB_DIR_PATH = "../baseline/db"
embeddings = OpenAIEmbeddings(model='text-embedding-3-large',api_key=config["API_KEY"])
db = DBClient(embedding_model=embeddings,db_dir_path=DB_DIR_PATH,doc_dir_path="../data")
mr = MultiStageRetriever(pathToDB=DB_DIR_PATH,specCollectionName=SPEC_COLL_NAME,reasonCollectionName=TDOC_COLL_NAME,diffCollectionName=DIFF_COLL_NAME)

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

def count_hit_rate_with_retrieval(chunks_in_file,org_chunk):
    """Calculates precision and recall, based on all of the `sections` retrieved by the retriever. 
    So tp,tn,fp,fn is calculated based on the number of sections not the number of chunks who meet criteria"""
    true_refs = get_refs_without_tables(RE.runREWithDocList([org_chunk]))
    true_refs = set(RE.extractClauseNumbersOfSrc(true_refs))

    #ensure that only clauses that can be inthe document are accessed
    true_refs,all_sections_in_org_file = get_all_existing_sections(true_refs,chunks_in_file)
    #print(f"true_refs:{true_refs}, all_refs are {all_sections_in_org_file}")

    retriever_result = mr.invoke(org_chunk.page_content)
    org_docs,add_docs = retriever_result.firstOrderSpecDocs,retriever_result.secondOrderSpecDocs
    org_docs += add_docs
    #check the sections of the org_docs.  
    retriever_refs = set()
    for doc in org_docs:
        retriever_refs.add(doc.metadata["section"])

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

def process_file(file,chunks):
    mr.constructRetriever(db, selected_docs=[clean_file_name(file)])
    chunks_with_refs, _ = get_chunks_with_refs(chunks)

    results = {}
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(count_hit_rate_with_retrieval, chunks, chunk): chunk for chunk in chunks_with_refs}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Chunks in {file}"):
            res = future.result()
            results.update(res)

    return get_avg_scores_for_file(file, results)

if __name__ == "__main__":
    ## Setup classes

    ## for each doc, get chunks and see hit rate
    file_list = ["../data/38211-i70.docx","../data/38212-i70.docx","../data/38213-i70.docx","../data/38214-i70.docx"]
    all_chunks = getFullSectionChunks(file_list)

    # group by file
    chunks_by_file = {}
    for chunk in all_chunks:
        src = chunk.metadata["source"]
        chunks_by_file.setdefault(src, []).append(chunk)


    final_results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:  # 4 files at once
        futures = [pool.submit(process_file, file, chunks) for file, chunks in chunks_by_file.items()]
        for fut in  tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            res = fut.result()
            # res looks like {"file_name": ..., "avg_precision": ..., "avg_recall": ..., "avg_f1": ...}
            final_results[res["file_name"]] = {
                "precision": res["avg_precision"],
                "recall": res["avg_recall"],
                "f1": res["avg_f1"]
            }

    # Output as JSON
    print(json.dumps(final_results, indent=2))
