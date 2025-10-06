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
import argparse
from SystemModels import BaseSystemModel, ControllerSystemModel,Chat3GPPAnalogueModel
#get the retriever and client

RE = ReferenceExtractor()
DB_DIR_PATH = "../baseline/db"

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

def get_all_existing_sections(ref_set:set,docID_to_file:dict,file_to_sections:dict)->Tuple[set,set]:
    true_refs = set()
    for ref in ref_set:
        file = docID_to_file.get(ref[0],None)
        if file:
            if ref[1] in file_to_sections[file]:
                true_refs.add(ref)
    return true_refs

def get_docid_and_section_ref_pairs(org_chunk):
    """Given a chunk, return all (docID,section) pairs that it references."""
    refs = get_refs_without_tables(RE.runREWithDocList(docs=[org_chunk]))
    ref_pairs = set()
    for ref in refs:
        section_name = RE.extractClauseNumbersFromString(ref.reference)[0]
        if ref.src == RE.getSRCDOC():
            docId = org_chunk.metadata["docID"]
        else:
            docId = ref.src
        ref_pairs.add((docId,section_name))
    return ref_pairs

def process_document_into_dict(doc)->dict:
    return {"metadata":{'section':doc.metadata["section"],"docID":doc.metadata["docID"]},"page_content":doc.page_content}

def count_hit_rate_with_retrieval(chunks_in_file,org_chunk, system:BaseSystemModel,current_file:str,docID_to_file,file_to_sections)->dict:
    """Calculates precision and recall, based on all of the `sections` retrieved by the retriever. 
    So tp,tn,fp,fn is calculated based on the number of sections not the number of chunks who meet criteria"""
    true_refs = get_refs_without_tables(RE.runREWithDocList([org_chunk]))
    # todo: rework tre_pairs definition and the function to get_all_existing_sections 
    #true_pairs = set((org_chunk.metadata["docID"],sec)for sec in RE.extractClauseNumbersOfSrc(true_refs))
    true_pairs = get_docid_and_section_ref_pairs(org_chunk)


    #ensure that only clauses that can be inthe document are accessed
    true_pairs = get_all_existing_sections(true_pairs,docID_to_file=docID_to_file,file_to_sections=file_to_sections)
    #print(f"true_refs:{true_refs}, all_refs are {all_sections_in_org_file}")

    _,org_docs = system.get_only_retrieval_results(org_chunk.page_content)

    retrieval_log = []
    for doc in org_docs:
        retrieval_log.append({
            "org_doc": process_document_into_dict(org_chunk),
            "ref_doc": process_document_into_dict(doc)
        })

    #check the sections of the org_docs.  
    retriever_pairs = set()
    for doc in org_docs:
        retriever_pairs.add((doc.metadata["docID"],doc.metadata["section"]))

    tp = len(retriever_pairs.intersection(true_pairs))
    fp = len(retriever_pairs.difference(true_pairs))

    fn =len(true_pairs.difference(retriever_pairs))

    print(f"File: {current_file}, TP: {tp}, FP: {fp}, FN: {fn}")

    precision = tp/(tp+fp) if tp+fp !=0 else 0.0
    recall = tp/(tp+fn) if tp+fn !=0 else 0.0
    f1_score = (2*precision*recall)/(precision+recall) if precision and recall else 0.0

    return {org_chunk.metadata["section"]:{"tp":tp,"fp":fp,"fn":fn}}, {f"{org_chunk.metadata['docID']}::{org_chunk.metadata['section']}": retrieval_log}

def get_avg_scores_for_file(file_name:str,results_dict:dict)->dict:
    tot_tp,tot_fp,tot_fn = 0,0,0
    n = max(len(results_dict),1)
    for section in results_dict.keys():
        tot_tp += results_dict[section]["tp"]
        tot_fp += results_dict[section]["fp"]
        tot_fn += results_dict[section]["fn"]
    return {"file_name":clean_file_name(file_name),"tot_tp":tot_tp,"tot_fp":tot_fp,"tot_fn":tot_fn,"num_chunks_with_refs":n}

def process_file(file,chunks,system:BaseSystemModel,docID_to_file,file_to_sections)->dict:
    chunks_with_refs, _ = get_chunks_with_refs(chunks)

    results = {}
    local_cache = {}
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(count_hit_rate_with_retrieval, chunks, chunk,system, file,docID_to_file,file_to_sections): chunk for chunk in chunks_with_refs}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Chunks in {file}"):
            res,cache_updates = future.result()
            results.update(res)
            for k, v in cache_updates.items():
                local_cache.setdefault(k, []).extend(v)

    return get_avg_scores_for_file(file, results),local_cache

if __name__ == "__main__":
    ## Setup classes
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--use-system',action='store_true',help="pass this argument to use deepspecs")
    argparser.add_argument('--use-3gpp',action='store_true',help='pass this argument to use chat3gpp analogue')
    argparser.add_argument('--output-path','-o',type=str,default='misalignment_results.json')
    args = argparser.parse_args()

    if args.use_3gpp and args.use_system:
        raise Exception("Error: Can't use multiple models at once")

    if args.use_system:
        system = ControllerSystemModel(isDBInitialized=True,doc_dir_path="../data",db_dir_path="../baseline/db")
        print("USING SYSTEM")
    elif args.use_3gpp:
        system = Chat3GPPAnalogueModel("../testdb",isEvol=False)
        print("USING Chat3gpp")

    ## for each doc, get chunks and see hit rate
    file_list = ["../dataformicrobenchmark/38181-i70.docx","../dataformicrobenchmark/38201-i00.docx","../dataformicrobenchmark/38202-i40.docx",\
                 "../dataformicrobenchmark/38211-i70.docx","../dataformicrobenchmark/38212-i70.docx", "../dataformicrobenchmark/38213-i70.docx",\
                    "../dataformicrobenchmark/38214-i70.docx", "../dataformicrobenchmark/38215-i40.docx","../dataformicrobenchmark/38300-i70.docx",\
                    "../dataformicrobenchmark/38304-i40.docx"    ]
    all_chunks = getFullSectionChunks(file_list)

    # group by file
    chunks_by_file = {}
    # file to sections will be used to ensure that we only consider sections that exist in the document
    file_to_sections = {clean_file_name(file):set() for file in file_list}
    docID_to_file = {}
    retrieval_cache = {}

    for chunk in all_chunks:
        src = chunk.metadata["source"]
        chunks_by_file.setdefault(src, []).append(chunk)
        file_to_sections[clean_file_name(src)].add(chunk.metadata["section"])
        docID_to_file[chunk.metadata["docID"]] = clean_file_name(src)


    final_results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:  # 4 files at once
        futures = [pool.submit(process_file, file, chunks,system,docID_to_file,file_to_sections) for file, chunks in chunks_by_file.items()]
        for fut in  tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            res,local_cache = fut.result()
            # res looks like {"file_name": ..., "avg_precision": ..., "avg_recall": ..., "avg_f1": ...}
            final_results[res["file_name"]] = {
                "tot_tp": res["tot_tp"],
                "tot_fp": res["tot_fp"],
                "tot_fn": res["tot_fn"],
                "num_chunks_with_refs": res["num_chunks_with_refs"]
            }
            for k, v in local_cache.items():
                retrieval_cache.setdefault(k, []).extend(v)

    # Output as JSON
    with open(args.output_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    
    with open("retrieval_cache.json", "w") as f:
        json.dump(retrieval_cache, f, indent=4)
