import sys
sys.path.append("../")
from ReferenceExtractor import ReferenceExtractor
from utils import RefObj
from langchain_core.documents import Document
from MetadataAwareChunker import getFullSectionChunks
from typing import Tuple

RE =ReferenceExtractor()

SECTIONS_TO_BE_DROPPED = ["Foreword", "N/A", "Annex A (informative):\nChange history"]
SRC_DOC = "Current_Doc"

def filter_chunks(chunks_with_metadata:list[Document]):
    """Filter out all the document chunks from the blacklisted sections"""
    chunks = []
    for chunk in chunks_with_metadata:
        if chunk.metadata["section"].strip() not in SECTIONS_TO_BE_DROPPED and "Annex" not in chunk.metadata["section"]:
            chunks.append(chunk)
    return chunks

def filter_refs(extRefs:list[RefObj],curr_section_name:str):
    """This will remove all duplicate refObjs and all refObjs that fit a certain criteria.
    For now, we want to remove any ref that is of the same section name as @curr_section_name and same doc"""
    extRefs = list(set(extRefs))
    filtered_refs = []
    for extRef in extRefs:
        if extRef.reference == curr_section_name and extRef.src == SRC_DOC:
            pass
        else:
            filtered_refs.append(extRef)
    return filtered_refs

def get_intrarelation_measure(chunks_with_metadata:list[Document])->Tuple[dict,dict]:
    """How many of the chunks of the document have external references. 
    Avg external reference number."""
    num_chunks_with_internal_refs = 0
    num_chunks_with_external_refs = 0
    num_chunks_with_any_refs = 0
    avg_num_references = 0 #avg number of all refs across all the chunks
    avg_per_chunk_with_refs = 0 #avg number of all refs across only chunks which have refs
    section_to_ref_num = {}

    for chunk in chunks_with_metadata:
        curr_section_name = chunk.metadata["section"]
        allRefs:list[RefObj] = filter_refs(RE.runREWithDocList(docs=[chunk]),curr_section_name)
        intraDocRefs = RE.extractClauseNumbersOfSrc(allRefs)

        if len(intraDocRefs) >0 or (len(allRefs) - len(intraDocRefs)) > 0:
            num_chunks_with_any_refs +=1
        if len(intraDocRefs) >0:
            num_chunks_with_internal_refs +=1
        if len(allRefs) - len(intraDocRefs) >0:
            num_chunks_with_external_refs +=1

        section_to_ref_num[curr_section_name] = {'num_of_other_section_refs':len(intraDocRefs),'num_of_other_doc_refs':len(allRefs)-len(intraDocRefs)}
        avg_num_references += len(allRefs)
        if len(allRefs) > 0:
            avg_per_chunk_with_refs += len(allRefs)

    avg_num_references /= len(chunks_with_metadata)
    avg_per_chunk_with_refs /= num_chunks_with_any_refs
    return {"avg_num_references":avg_num_references,"avg_per_chunk_with_refs":avg_per_chunk_with_refs,"num_chunks_with_any_refs":num_chunks_with_any_refs,"num_chunks_with_internal_refs":num_chunks_with_internal_refs,"num_chunks_with_external_refs":num_chunks_with_external_refs},section_to_ref_num

if __name__ == "__main__":
    file_list = ["../data/38214-i60.docx"]
    chunks_with_metadata = filter_chunks(getFullSectionChunks(file_list))
    for chunk in chunks_with_metadata:
        print(chunk.metadata["section"])
    print(len(chunks_with_metadata))
    result,breakdown = get_intrarelation_measure(chunks_with_metadata)
    #print(breakdown)
    print(f"{result}")
    

