import sys
sys.path.append("..")
import os
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from utils import getAllFilesInDirMatchingFormat
from MetadataAwareChunker import getFullSectionChunks,addExtraDocumentWideMetadataForReason
from DBClient import DBClient
from settings import config
from ChangeTracker import ChangeTracker, get_empty_document

DIFF_DOC_DIR = "../testchange"

BLACKLISTED_SECTIONS = ["3.1","3.2","3.3","3","1","2","Foreword","N/A"]

#get diffs from docs in testchange.

def get_diff_list():
    """Returns: a list of strings where each string represents a change made"""

    diff_list = []

    versionToSectionToChunk = {}
    versionToMetadata = {}
    file_list = getAllFilesInDirMatchingFormat(DIFF_DOC_DIR)
    for file in file_list:
        chunks = getFullSectionChunks([os.path.join(DIFF_DOC_DIR,file)],addExtraDocumentWideMetadataForReason)
        version = chunks[0].metadata["version"]
        versionToSectionToChunk[version] = {}
        versionToMetadata[version] = chunks[0].metadata
        for chunk in chunks:
            versionToSectionToChunk[version][chunk.metadata["section"]] = chunk
        
    for fromVersion in versionToSectionToChunk.keys():
        for toVersion in versionToSectionToChunk.keys():
            if ChangeTracker.areAdjacentVersions(fromVersion=fromVersion,toVersion=toVersion):
                for section in versionToSectionToChunk[toVersion].keys():
                    if section in BLACKLISTED_SECTIONS:
                        continue
                    defaultFromChunk = get_empty_document(metadata={**versionToMetadata[fromVersion],'section':section})
                    fromChunk = versionToSectionToChunk[fromVersion].get(section,defaultFromChunk)
                    toChunk = versionToSectionToChunk[toVersion][section]
                    
                    change_obj = ChangeTracker.getChanges(fromChunk,toChunk)
                    for change in change_obj['changes']['add']:
                        diff_list.append({'metadata':change_obj['metadata'],'text':change})
    return diff_list

    

#ask llm to come up with a question from the diff. Question should be about why that change was introduced
question_gen_template = '''Here is a chunk of text that represents text that was added between two versions of a 3GPP Technical Specification.
Come up with a question that asks about why this change is made or why the new feature in the change is supported. 
The change: {diff_text}
Instructions:
1. The question should not be answerable with only the text of the change.
2. Respond only with the question.'''
def get_response_question_generation(client,diff_text,seed):
    response_prompt = question_gen_template.format(diff_text=diff_text)
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


# for each diff, grab the text of the add change and give it to the llm.
def get_question_from_diff(client,diff:dict,seed, max_retries=3, delay=3):
    """Encapsulates the logic for processing a single item."""
    for attempt in range(1, max_retries + 1):
        try:
            gpt_response = get_response_question_generation(client,diff["text"], seed)
            return {
                'question': gpt_response,
                'fromVersion': diff['metadata']['from_version'],
                'toVersion': diff['metadata']['to_version'],
                'fromTimestamp':diff['metadata']['fromTimestamp'],
                'toTimestamp': diff['metadata']['toTimestamp'],
                'srcContentForQuestion':diff['text'],
                'section':diff['metadata']['section'],
            }
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                print(f"Max retries reached, skipping this item.")
                return {
                    'question': None,
                    'fromVersion': diff['metadata']['from_version'],
                    'toVersion': diff['metadata']['to_version'],
                    'fromTimestamp':diff['metadata']['fromTimestamp'],
                    'toTimestamp': diff['metadata']['toTimestamp'],
                    'srcContentForQuestion':diff['text'],
                    'section':diff['metadata']['section'],
                }
# then, use this created question to query the discussion db
# grab the context from the discussion db

def buildMetadataFilterFromKeywords(keywords:list[str]):
    filters = []
    for keyword in keywords:
        filters.append({'$contains':keyword})
    if filters == []:
        return {}
    if len(filters) == 1:
        return filters[0]
    return {'$or':filters}

def get_related_tdoc(client,collection,question,seed):
    gpt_response = get_keywords_from_question(client,question,seed)
    keywords = gpt_response.strip("[]").split(",")
    print(f"for question {question}, keywords are {keywords}")
    metadata_filter = buildMetadataFilterFromKeywords(keywords)

    #related_tdocs = db.vector_db.similarity_search(question,3,filter=metadata_filter)
    related_tdocs = collection.query(
    query_texts=[question],
    n_results=3,
    where_document=metadata_filter 
)
    if related_tdocs == []:
        return ""
    return "\n".join([tdoc for tdoc in related_tdocs["documents"][0]])

# filter to see if tdocs are actually relevant to the questions   
docs_relevant_template = '''Extract 3 keywords from the question given below.
<question>
{question}
</question>
Instructions:
1. Return only a list of the form [<keyword1>,<keyword2>,<keyword3>].
2. Acronyms are usually keywords.
3. Keywords are usually one word.
4. Do not change how the keyword is written in the question.
5. Keywords are usually objects of the sentence.''' 
def get_keywords_from_question(client,question,seed):
    response_prompt = docs_relevant_template.format(question=question)
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

# pass this context into second prompt, along with question. Ask it to come up with a ground truth answer
answer_gen_template = '''Answer the following question in 200 words with reference to the provided context:
<context>
{context}
</context>
Question: {question}
'''
def get_response_answer_gen(client,question,context,seed):
    response_prompt = answer_gen_template.format(question=question,context=context)
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

def get_answer_to_question(client,question_obj,context,seed,max_retries=3,delay=3):
    """Encapsulates the logic for processing a single item."""
    question = question_obj["question"]
    for attempt in range(1, max_retries + 1):
        try:
            gpt_response = get_response_answer_gen(client,question,context,seed)
            return {
                **question_obj,
                'ground_truth':gpt_response,
                'context_for_answer': context
            }
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                print(f"Max retries reached, skipping this item.")
                return {
                    **question_obj,
                    'ground_truth':None
                }

if __name__ == "__main__":
    output_path = "./results2.json"

    #embeddings = OpenAIEmbeddings(model='text-embedding-3-large',api_key=config["API_KEY"])
    #db = DBClient(embedding_model=embeddings,collection_name=config["TDOC_COLL_NAME"],db_dir_path=os.path.join("..",config["CHROMA_DIR"]),doc_dir_path=os.path.join("..","reasoning"))

    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=config["API_KEY"],
        model_name="text-embedding-3-large"
    )
    db = chromadb.PersistentClient(path=os.path.join("..",config["CHROMA_DIR"]))
    collection = db.get_collection(name=config["TDOC_COLL_NAME"],embedding_function=embedding_fn)

    client = OpenAI(
        api_key=config["API_KEY"],
        timeout=60,
    )

    diff_list = get_diff_list()[:10]
    question_objs = [None] * len(diff_list)
    
    # Use ThreadPoolExecutor to process up to 60 items at once
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit each item and store a mapping from Future -> index
        future_to_index = {}
        for i, diff in enumerate(diff_list):
            future = executor.submit(get_question_from_diff, client, diff, seed=0)
            future_to_index[future] = i

        # As each future finishes, insert its result into `results`
        n_finished = 0
        for future in tqdm(as_completed(future_to_index),
                           total=len(future_to_index),
                           desc="Processing"):
            i = future_to_index[future]
            result = future.result()
            question_objs[i] = result
            n_finished += 1

        
    # Write questions to disk
    #with open(output_path, 'w') as f:
        #json.dump(question_objs, f, indent=4)
    
######################################################################################################################

    answer_objs = [None] * len(question_objs)
    
    # Use ThreadPoolExecutor to process up to 60 items at once
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit each item and store a mapping from Future -> index
        future_to_index = {}
        for i, question_obj in enumerate(question_objs):
            context = get_related_tdoc(client,collection,question_obj["question"],seed=0)
            if context == "":
                continue
            future = executor.submit(get_answer_to_question, client, question_obj,context, seed=0)
            future_to_index[future] = i

        # As each future finishes, insert its result into `results`
        n_finished = 0
        for future in tqdm(as_completed(future_to_index),
                           total=len(future_to_index),
                           desc="Processing"):
            i = future_to_index[future]
            result = future.result()
            answer_objs[i] = result
            n_finished += 1

            # Write partial results to disk after each item completes
            if n_finished == 1 or n_finished % 100 == 0:
                with open(output_path, 'w') as f:
                    json.dump(answer_objs, f, indent=4)
        
    # Write final results to disk
    with open(output_path, 'w') as f:
        json.dump(answer_objs, f, indent=4)

    
