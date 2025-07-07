import sys
sys.path.append("..")
from ReferenceExtractor import ReferenceExtractor
from MetadataAwareChunker import getFullSectionChunks
from settings import config
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from openai import OpenAI


RE = ReferenceExtractor()

response_template = '''Here are two different chunks of text from a 3GPP specification. Each of these chunks represents a different clause/subclause in the text. The two chunks reference each other.
This means that there are parts of chunk1 that need context from chunk2 to fully be understood and vice versa.
Come up with a question that would require the context from both chunks to answer, and the answer to that question.
chunk1:{chunk1content} which is from clause {section1}.
chunk2:{chunk2content} which is from clause {section2}.
Instructions:
1. Do not use the terms chunk1 or chunk2 in the generated question
2. The question generated MUST ask about a concept that is split between the two chunks/sections.
3. The question generated MUST require bridging the reference between sections
4. The question must require a logical connection between two points not found in the same chunk
5. The question MUST be unanswerable using only chunk1 or only chunk2 in isolation
6. Ensure the answer is complete and clearly grounded in content from both chunks.
7. Respond only with a json of the form {{"question":"...","answer":"..." }} where question is the generated question and answer is the answer to that question. There should be no ``` or word json in the response, only the dictionary'''
def get_response(client,chunk1,chunk2, seed):
    chunk1content,section1 = chunk1.page_content,chunk1.metadata["section"]
    chunk2content,section2 = chunk2.page_content, chunk2.metadata["section"]

    response_prompt = response_template.format(chunk1content=chunk1content,section1=section1,chunk2content=chunk2content,section2=section2)
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
                'ground_truth': response_obj["answer"]
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

# get chunks that refer to one another
def get_double_ref_pairs(chunks)->list[tuple]:
    edges = []
    ref_section_to_chunk = {chunk.metadata["section"]:chunk for chunk in chunks}
    for chunk in chunks:
        true_refs = RE.runREWithDocList([chunk])
        true_refs = set(RE.extractClauseNumbersOfSrc(true_refs))

        for ref in true_refs:
            if ref_section_to_chunk.get(ref,None):
                edges.append((chunk,ref_section_to_chunk[ref]))
    return edges

if __name__ == "__main__":
    file = "../data/38211-i60.docx"
    results = {}

    chunks = getFullSectionChunks([file])
    edges = get_double_ref_pairs(chunks)

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
        
