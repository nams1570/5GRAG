# Overview
This is a 5G expert system. It uses retrieval augmented generation to augment the Q&A capabilities of an LLM. 


# Before you begin
Make sure you have a `settings.yml` file in the same directory as `settings.py`.
It should have these vars:
1. `API_KEY` (str): Your openai api key.
2. `DOC_DIR` (str): The name of the directory where the files fetched by the AutoFetcher will be deposited.
3. `MODEL_NAME` (str). The openai model that will be used as the core of the retrieval chain. By default, try "gpt-5-mini"
4. `NUM_EXTRA_DOCS` -> the number of additional docs to retrieve per run. NOT depth, closer to top k
5. `CHROMA_DIR` -> The directory the chromadb sqlite db will be stored
6. `SPEC_COLL_NAME`: "context" 
7. `TDOC_COLL_NAME`: "reason"
8. `DIFF_COLL_NAME`: "diff"
9. `IS_SMART_RETRIEVAL`: (boolean) indicates whether or not smart retrieval/ deep context is turned on. By default, this should be "true"
10. `NUM_DOCS_INITIAL_RETRIEVAL`: (int) the number of documents retrieved by the first round (non deep context) of retrieval from the specDB
11. `NUM_REASONING_DOCS_TO_RETRIEVE`: (int) max number of documents retrieved from TdocDB
12. `DEPTH`: (int) How many iterations of the secondary context retrieval you want to go to. By default, this should be 1.

# Databases and some elaboration
The context is stored in 3 chromadb collections: `specDB` -which holds the technical spec chunks, `changeDB` - which holds the diffs between adjacent versions of the same spec, and `TdocDB` - which holds context from change requests.  These three collections must be stored in the same chromadb sqlite database. The `x_COLL_NAME` fields in the settings.yml file represent the names of the aforementioned collections. Note that if you want to reuse the same collections and have them findable by the system, **you must keep the collection names consistent between runs**.

# How to set up the dbs/ collections
There are three scripts that should be run to initialize your databases/collections:
1. `ConstructSpecDB.py`: This constructs the `specDB` collection.
2. `ConstructDiffDB.py`: This constructs the `changeDB` collection.
3. `ConstructReasonDB.py`: This constructs the `TdocDB`collection.
Note that in each of these files, you will have to update the `DB_DIR_PATH` and the `DOC_DIR_PATH` variables. 

The `DB_DIR_PATH` refers to the folder where the chromadb sqlite instance will be kept and where all your collections should go. **Make sure that the `DB_DIR_PATH` is consistent across all 3 scripts**

The `DOC_DIR_PATH` is where the documents to be parsed will be read from. Note that the folders are not parsed recursively: they should just contain the docx files. For `TdocDB`, make sure that the folder contains only change requests. For `specDB` and `changeDB`, make sure that the folder contains technical specifications of 3gpp.

# Running the system
Do `python frontend.py`. In your terminal, an ip address will be exposed.
Navigate to this ip address and you should see the gradio interface. If your chroma databases are all set up, you should just be able to type your questions into the chat and see the retrieved context and the answer.


# Dependencies
Please see the requirements.txt!

# Things of note:
The DB is persistent so between runs you access the same databases.