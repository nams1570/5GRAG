# Overview
This is a 5G expert system. It uses retrieval augmented generation to augment the Q&A capabilities of an LLM. It is not conversational. Each question is independent of the previous one, and we do not maintain history.


# Before you begin
Make sure you have a `settings.yml` file in the same directory as `settings.py`.
It should have these vars:
1. `API_KEY` (str): Your openai api key.
2. `DOC_DIR` (str): The default directory that any DBClient instance is going to try to read from when adding files to a collection. 
3. `MODEL_NAME` (str). The openai model that will be used as the core of the retrieval chain. By default, try "gpt-5-mini"
4. `NUM_EXTRA_DOCS` -> the number of additional docs to retrieve per run. NOT depth, closer to top k. This is the total number of additional docs to retrieve from specDB summed across all depth levels.
5. `CHROMA_DIR` -> The directory the chromadb sqlite db will be stored
6. `IS_SMART_RETRIEVAL`: (boolean) indicates whether or not smart retrieval/ deep context is turned on. By default, this should be "true"
7. `NUM_DOCS_INITIAL_RETRIEVAL`: (int) the number of documents retrieved by the first round (non deep context) of retrieval from the specDB
8. `NUM_REASONING_DOCS_TO_RETRIEVE`: (int) max number of documents retrieved from TdocDB
9. `DEPTH`: (int) How many iterations of the secondary context retrieval you want to go to. By default, this should be 1.

# Databases and some elaboration
The context is stored in 3 chromadb collections: `specDB` -which holds the technical spec chunks, `changeDB` - which holds the diffs between adjacent versions of the same spec, and `TdocDB` - which holds context from change requests.  These three collections must be stored in the same chromadb sqlite database. The `x_COLL_NAME` fields mentioned in other files represent the names of the aforementioned collections. Note that if you want to reuse the same collections and have them findable by the system, **you must keep the collection names consistent between runs**.

# How to set up the dbs/ collections
There are three scripts that should be run to initialize your databases/collections:
1. `ConstructSpecDB.py`: This constructs the `specDB` collection.
2. `ConstructDiffDB.py`: This constructs the `changeDB` collection.
3. `ConstructReasonDB.py`: This constructs the `TdocDB`collection.
Note that in each of these files, you will have to update the `DB_DIR_PATH` and the `DOC_DIR_PATH` variables. 

The `DB_DIR_PATH` refers to the folder where the chromadb sqlite instance will be kept and where all your collections should go. **Make sure that the `DB_DIR_PATH` is consistent across all 3 scripts**

The `DOC_DIR_PATH` is where the documents to be parsed will be read from. Note that the folders are not parsed recursively: they should just contain the docx files. For `TdocDB`, make sure that the folder contains only change requests. For `specDB` and `changeDB`, make sure that the folder contains technical specifications of 3gpp.

# If you don't want to set up your own databases
[Here](https://ucla.box.com/s/q9wxe7r06wq7uecr12c7g0lrbrnzn3p3) is a link to a zip of a chromadb database, with collections representing `specDB`, `changeDB`, and `tdocDB`. It contains around 1000 CRs, over 2200 Technical Specifications, and about 50 documents worth of diffs. The Technical Specifications are from release 17 and 18 ranging from docIDs 21.101 to 55.919.  The CRs correspond to the technical specs with docIDs ranging from 37.213 to 38.901

To use this db for a simple plug and play, copy the `db` file into the `baseline` directory. That is, there should be a `baseline/db/` subdir inside of which the collections and sqlite instance should appear. The collection names should be the same as in the `CollectionNames.py` file, so you will probably not have to make any changes there.

# Running the system

## Running the composite system
Do `python frontend.py`. In your terminal, an ip address will be exposed.
Navigate to this ip address and you should see the gradio interface. If your chroma databases are all set up, you should just be able to type your questions into the chat and see the retrieved context and the answer.

## Strictly setting up the deepspecs server
You can also just set up the deepspecs controller as a server with FastAPI waiting for responses.
This way you can query it with any client.
Run `fastapi dev ./ds_server.py` to get it going.
Then, query the API with the post requests to either directly access the diff db or to post your question and get an answer.

## Running it in a container
We have a Dockerfile that will transport the relevant deepspecs files to the container, and set it up to run the client in production.

Use `docker build -t myimage .` while in the directory with the Dockerfile to build an image called myimage. This may take a while, but only needs to be done once.

Use `docker run --rm --memory="12g" --memory-swap="16g" -v "<absolute_path_to_db>:/baseline/db" -d --name mycontainer -p 8000:8000 myimage` to run the container in detached mode. 
This command maps the chroma db on your local machine to the chroma db in the container, so replace `/baseline/db` with the correct path to the db as per your settings. 
8000:8000 maps the port 8000 of the container to the port 8000 of your machine. Note that the dockerfile command exposes port 8000 as the port of the server where the requests are served.

## Potential Docker memory concerns
Due to the large size of the db, we set the memory limit to 12 GiB with a memory swap even larger. The first request that is served will be slow as the db is being pulled into memory via a volume mont. Subsequent requests will be faster. Note that if you are plugging in a larger chroma db, you may need more memory.

If you are using wsl on Windows, unless specified otherwise in your Docker desktop settings, your max memory remit is set by the `.wslconfig` file in your `C:\%user%` directory. Here is a template you can follow in your .wslconfig file: 
```
[wsl2]
memory=16GB        # limit WSL2 VM to 16 GB
processors=4
swap=8GB
localhostForwarding=true
```
Change as needed. You can debug the memory usage of your container while it is running using `docker stats --no-stream`.

# Dependencies
Please see the requirements.txt!

# Things of note:
The DB is persistent so between runs you access the same databases.