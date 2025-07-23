import sys
sys.path.append("..")
from ReferenceExtractor import ReferenceExtractor
from MetadataAwareChunker import getFullSectionChunks
from settings import config
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import argparse
from openai import OpenAI

BLACKLISTED_SECTIONS = ["3.1","3.2","3.3","3","1","2","Foreword","6.1","6.2","6.3","6.4","6.5","4.1","4.2","4.3","4.2.1","4.2.2","4.3.1","4.3.2","4.4"]

RE = ReferenceExtractor()

response_template = '''Here are two different chunks of text from a 3GPP specification. Each of these chunks represents a different clause/subclause in the text. The two chunks reference each other.
This means that there are parts of chunk1 that need context from chunk2 to fully be understood and vice versa.
Come up with a question that would require the context from both chunks to answer, and the answer to that question.
From clause {section1} in document {docID1}, we have chunk1:{chunk1content}.\n\n
From clause {section2} in document {docID1}, we have chunk2:{chunk2content}.\n\n
Instructions:
1. Do not use the terms chunk1 or chunk2 in the generated question
2. The question generated MUST ask about a concept that is split between the two chunks/sections.
3. The question generated MUST require bridging the reference between sections
4. The question must require a logical connection between two points not found in the same chunk
5. The question MUST be primarily based off the part in chunk1 where chunk2 is mentioned (via its clause number)
6. The question MUST be unanswerable using only chunk1 or only chunk2 in isolation
7. Ensure the answer is complete and clearly grounded in content from both chunks.
8. Respond only with a json of the form {{"question":"...","answer":"...", "reason":"... }} where question is the generated question and answer is the answer to that question and reason is the reasoning behind why the question was created. There should be no ``` or word json in the response, only the dictionary'''
def get_response(client,chunk1,chunk2, seed):
    chunk1content,section1,docID1 = chunk1.page_content,chunk1.metadata["section"],chunk1.metadata["docID"]
    chunk2content,section2, docID2 = chunk2.page_content, chunk2.metadata["section"],chunk2.metadata["docID"]

    response_prompt = response_template.format(chunk1content=chunk1content,section1=section1,chunk2content=chunk2content,section2=section2,docID1=docID1,docID2=docID2)
    print(response_prompt)
    response = client.chat.completions.create(
        model=config["MODEL_NAME"],
        messages=[
            {
                'role': 'user',
                'content': response_prompt
            }
        ],
        seed=seed,
    )
    return response.choices[0].message.content

def process_item(client, chunk1,chunk2, seed, max_retries=3, delay=3):
    """Encapsulates the logic for processing a single item."""
    for attempt in range(1, max_retries + 1):
        try:
            gpt_response = get_response(client,chunk1,chunk2, seed)
            response_obj = json.loads(gpt_response)
            return {
                'question': response_obj["question"],
                'ground_truth': response_obj["answer"],
                'primary_chunk_section': chunk1.metadata["section"],
                'primary_chunk_text': chunk1.page_content,
                'primary_chunk_doc': chunk1.metadata["docID"],
                'secondary_chunk_section':chunk2.metadata["section"],
                'secondary_chunk_text': chunk2.page_content,
                'secondary_chunk_doc': chunk2.metadata["docID"]
            }
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                print(f"Max retries reached, skipping this item.")
                return {
                    'question': None,
                    'answer': None
                }

# get chunks that refer to one another, internal references
def get_double_ref_pairs(chunks)->list[tuple]:
    edges = []
    ref_section_to_chunk = {chunk.metadata["section"]:chunk for chunk in chunks}
    for chunk in chunks:
        if chunk.metadata['section'] in BLACKLISTED_SECTIONS:
            continue
        true_refs = RE.runREWithDocList([chunk])
        true_refs = set(RE.extractClauseNumbersOfSrc(true_refs)) - set(BLACKLISTED_SECTIONS)
        for ref in true_refs:
            if ref_section_to_chunk.get(ref,None) != None:
                    if chunk.page_content == "" or ref_section_to_chunk[ref].page_content == "":
                        continue
                    edges.append((chunk,ref_section_to_chunk[ref]))
    return edges    

def get_ext_ref_pairs(chunks,docIdToChunkMap):
    edges = []
    org_docid = chunks[0].metadata['docID']
    for chunk in chunks:
        if chunk.metadata['section'] in BLACKLISTED_SECTIONS:
            continue
        true_refs = RE.runREWithDocList([chunk])
        for ref in true_refs:
            section_names = RE.extractClauseNumbersFromString(ref.reference)
            if ref.src == RE.getSRCDOC():
                continue
            else:
                org_chunk = docIdToChunkMap[org_docid][chunk.metadata['section']]
                if docIdToChunkMap.get(ref.src,None) == None:
                    continue
                if docIdToChunkMap[ref.src].get(section_names[0],None) == None:
                    continue
                ext_chunk = docIdToChunkMap[ref.src][section_names[0]]
                edges.append((org_chunk,ext_chunk))
    return edges

if __name__ == "__main__":
    files = ["../data/38211-i70.docx","../data/38212-i70.docx","../data/38213-i70.docx","../data/38214-i70.docx","../data/38304-i40.docx","../data/38321-i60.docx"]
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--size',type=int,default=100)
    argparser.add_argument('--internal',action='store_true')
    args = argparser.parse_args()

    results = {}
    docIdToChunkMap = {}
    fileToChunks = {}

    for file in files:
        chunks = getFullSectionChunks([file])
        fileToChunks[file] = chunks
        docId = chunks[0].metadata["docID"]
        docIdToChunkMap[docId] = {chunk.metadata["section"]:chunk for chunk in chunks}

    edges = []

    for file in files:
        chunks = fileToChunks[file]
        if not args.internal:
            edges += get_ext_ref_pairs(chunks,docIdToChunkMap)
        
        else:
            edges += get_double_ref_pairs(chunks)
    
    edges = edges[:args.size]

    results = [None] * len(edges)

    output_path = "./results.json"

    client = OpenAI(
        api_key=config["API_KEY"],
        timeout=60,
    )

    # Use ThreadPoolExecutor to process up to 60 items at once
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit each item and store a mapping from Future -> index
        future_to_index = {}
        for i, edge in enumerate(edges):
            chunk1,chunk2 = edge
            future = executor.submit(process_item, client, chunk1,chunk2, seed=0)
            future_to_index[future] = i

        # As each future finishes, insert its result into `results`
        n_finished = 0
        for future in tqdm(as_completed(future_to_index),
                           total=len(future_to_index),
                           desc="Processing"):
            i = future_to_index[future]
            result = future.result()
            results[i] = result
            n_finished += 1

            # Write partial results to disk after each item completes
            if n_finished == 1 or n_finished % 100 == 0:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=4)
        
    # Write final results to disk
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        
