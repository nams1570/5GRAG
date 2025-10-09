import sys
sys.path.append("..")
from ReferenceExtractor import ReferenceExtractor
from MetadataAwareChunker import getFullSectionChunks,clean_file_name
from get_misalignment_score import get_chunks_with_refs, get_refs_without_tables, process_document_into_dict
import json 

RE = ReferenceExtractor()

if __name__ == "__main__":
    file_list = ["../dataformicrobenchmark/38181-i70.docx","../dataformicrobenchmark/38201-i00.docx","../dataformicrobenchmark/38202-i40.docx",\
                 "../dataformicrobenchmark/38211-i70.docx","../dataformicrobenchmark/38212-i70.docx", "../dataformicrobenchmark/38213-i70.docx",\
                    "../dataformicrobenchmark/38214-i70.docx", "../dataformicrobenchmark/38215-i40.docx","../dataformicrobenchmark/38300-i70.docx",\
                    "../dataformicrobenchmark/38304-i40.docx"    ]
    all_chunks = getFullSectionChunks(file_list)

    id_section_to_chunk = {}
    for chunk in all_chunks:
        id_section_to_chunk[(chunk.metadata["docID"],chunk.metadata["section"])] = chunk

    tot_num_of_references = 0


    print(f"Total chunks: {len(all_chunks)}")

    results = []
    chunks_with_refs, _ = get_chunks_with_refs(all_chunks)

    for chunk in chunks_with_refs:
        allRefs = set(get_refs_without_tables(RE.runREWithDocList(docs=[chunk])))
        for ref in allRefs:
            print(f"ref: {ref}")
            docID = ref.src if ref.src != RE.getSRCDOC() else chunk.metadata["docID"]
            section = RE.extractClauseNumbersFromString(ref.reference)[0]
            if (docID,section) in id_section_to_chunk:
                results.append({
                    "org_doc": process_document_into_dict(chunk),
                    "ref_doc": process_document_into_dict(id_section_to_chunk[(docID,section)])
                })
            else:
                pass
                # print(f"No match found for chunk {chunk.metadata['docID']} section {chunk.metadata['section']} referencing section {ref}")
        tot_num_of_references += len(allRefs)

    print(f"Total chunks with refs: {len(chunks_with_refs)}")
    print(f"Total references: {tot_num_of_references}")

    with open("all_chunk_ref_pairs.json", 'w') as f:
        json.dump(results, f, indent=4)