from pydoc import text
from zipfile import ZipFile
import os
import subprocess
import json
import pandas as pd
from markitdown import MarkItDown
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
from settings import config
import tiktoken
from datetime import datetime
from dateutil.relativedelta import relativedelta

###########################
## File Processing Tools ##
###########################

def unzipFile(filepath,dest_dir):
    """filepath must be an absolute path. 
    Downloads all files from the zip, and clears out the zip file"""
    with ZipFile(filepath,'r') as zf:
        zf.extractall(path=dest_dir)
    
    os.remove(filepath)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def convertAllDocToDocx(input_dir:str,output_dir:str=None):
    """Converts all the .doc files in `input_dir` to .docx, and then removes the old docs.
    the .docx files will be stored in `output_dir` which defaults to the same as `input_dir`"""
    if not output_dir:
        output_dir = input_dir
    for filename in os.listdir(input_dir):
        if filename.endswith(".doc"):
            filepath = os.path.join(input_dir,filename)
            subprocess.run(['soffice','--headless','--convert-to','docx','--outdir',output_dir,filepath])
            os.remove(filepath)

def getAllFilesInDirMatchingFormat(target_dir:str,accepted_extensions:list[str]=[".docx"])->list[str]:
    """Helper function to return the names of all the files in `@target_dir` that match any one of the `@accepted_extensions`"""
    file_list = set()
    for filename in os.listdir(target_dir):
        for extension in accepted_extensions:
            if extension in filename:
                file_list.add(filename)
    return list(file_list)

def convertJsonToCsv(input_filename:str,output_filename:str):
    with open(input_filename, "r") as read_file:
        data = json.load(read_file)
    df = pd.DataFrame(data)
    df.to_csv(output_filename)


##############################
### Chunking Tools Section ###
##############################

def getFirstPageOfDocxInMarkdown(filepath:str):
    """filepath must be an absolute path"""
    md = MarkItDown(enable_plugins=False) # Set to True to enable plugins
    try:
        result = md.convert(filepath)
    except:
        print(f"issue in file {filepath} with md conversion")
    return result.text_content[:500]

def getFirstTwoPagesOfDocxInMarkdown(filepath:str):
    """filepath must be an absolute path"""
    md = MarkItDown(enable_plugins=False) # Set to True to enable plugins
    try:
        result = md.convert(filepath)
    except:
        print(f"issue in file {filepath} with md conversion")
    return result.text_content[:9000]

class ChangeChunk(BaseModel):
    summary: str = Field(..., description="Change summary (bullet or sentence)")
    reason: str = Field("", description="Why the change was introduced")
    consequence: str = Field("", description="Impact if the change is not implemented")

class ChangeChunkList(BaseModel):
    changeChunks: List[ChangeChunk] = Field(..., description="List of change chunks extracted from the CR")

def getCRContentFromLLM(text_chunk:str)->list[ChangeChunk] | list[dict]:
    """Extract change chunks from a CR markdown text.

    Returns a list of validated ChangeChunk instances on success, otherwise a list containing a raw fallback dict.
    """
    client = OpenAI(api_key=config["API_KEY"])

    system_prompt = {
        "role": "system",
        "content": (
            "You are an expert at extracting structured change information from change request (CR) markdown text. "
            "Given the CR text, focus only on three sections: 'summary of changes', 'reason for change', and 'consequences if not approved'. "
            "For each change mentioned in the 'summary of changes' section produce one object with keys:\n"
            "  - summary: the change text (single bullet or sentence)\n"
            "  - reason: the portion(s) of 'reason for change' that explain why this change was introduced\n"
            "  - consequence: the portion(s) of 'consequences if not approved' that describe the impact if this change is not implemented\n"
            "Output ONLY a JSON array (no markdown, no extra commentary). If there are multiple bullets in a section, map them in order; if mapping is ambiguous, include best-effort contextual text."
        )
    }
    user_prompt = {"role": "user", "content": text_chunk}

    response = client.responses.parse(
        model=config["MODEL_NAME"],
        input=[system_prompt, user_prompt],
        text_format=ChangeChunkList
    )

    return [dict(chunk) for chunk in response.output_parsed.changeChunks]
    

def getMetadataFromLLM(text_chunk:str)->dict:
    """Passes text to the LLM which then parses it into metadata"""
    client = OpenAI(api_key=config["API_KEY"])

    response = client.responses.parse(
        model=config["MODEL_NAME"],
        input=[
            {
                "role":"system","content":"You are an expert at structured data extraction. You will be given unstructured text in markdown and must convert it into the given structure. \
                    Some rules to keep in mind:\
                    1. version can be extracted by looking for something of the form 'x.y.z' where x,y, and z are numbers. It is prefaced by V or the word Version: \
                        2. docID can be extracted by looking for something of the form 'az x.y' where x and y are numbers and az represents any two letters. This usually follows the phrase 3GPP. The docid is the x.y \
                            3. the timestamp is a date of the form YYYY-MM\
                                4. Release is a number usually prefaced by the word 'Release' \
                                    5. If you cannot instantiate a field, populate it with the empty string."
            },
            {"role":"user","content":text_chunk},
        ],
        text_format=DocumentWideMetadata 
    )
    return dict(response.output_parsed)

class DocIDFromText(BaseModel):
    docID: str = Field("", description="The extracted document identifier, or empty string if none found. Each docID must be x.y where x and y are numbers. No extra spaces or anything else")

class DocIDFromTextList(BaseModel):
    docIDs: List[DocIDFromText] = Field(..., description="List of extracted document identifiers")

def getDocIDFromText(text_chunk:str)->List[str]:
    """Extracts the docID from a given text chunk using the LLM."""
    client = OpenAI(api_key=config["API_KEY"])

    response = client.responses.parse(
        model=config["MODEL_NAME"],
        input=[
            {
                "role":"system","content":"You are an expert at extracting document identifiers from unstructured text. \
                    A document identifier (docID) typically follows the format 'az x.y' where 'az' represents any two letters and 'x.y' are numbers. It is possible for the az to be omitted \
                    This identifier is often found in proximity to the phrase '3GPP'. \
                    Your task is to identify and extract all the docIDs from the provided text. \
                    If no valid docID is present, return an empty string."
            },
            {"role":"user","content":text_chunk},
        ],
        text_format=DocIDFromTextList
    )
    return [docIDchunk.docID for docIDchunk in response.output_parsed.docIDs if docIDchunk.docID != ""]

######################
## Tokenizing Tools ##
######################

def getTokenCount(text:str,model_name:str,supressWarning:bool=True):
    """Use this to get a picture of how many tokens the @text contains."""
    if "gpt" in model_name or model_name=="text-embedding-3-large":
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except Exception as e:
            if not supressWarning:
                print(f"ran into exception {e} when tokenizing, defaulting to 4o mini's tokenizer")
            encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        finally:
            num_tokens = len(encoding.encode(text))
    else:
        raise Exception("Unsupported model type!")
    return num_tokens

####################
## Miscellaneous  ##
####################
def get_inclusive_tstmp_range(fromTimestamp:str,toTimestamp:str)->list[str]:
    """timestamps must be in YYYY-MM format. from <= to. 
    Returns an inclusive range of valid timestamps"""
    start = datetime.strptime(fromTimestamp, "%Y-%m")
    end = datetime.strptime(toTimestamp, "%Y-%m")
    
    if start > end:
        raise ValueError("fromTimestamp must be earlier than or equal to toTimestamp")

    result = []
    current = start
    while current <= end:
        result.append(current.strftime("%Y-%m"))
        current += relativedelta(months=1)

    return result

####################
## Object Classes ##
####################

class RefObj:
    def __init__(self,reference:str,src:str):
        self.reference = reference
        self.src = src
    
    def __repr__(self):
        return f'Reference: {self.reference}, Source Document: {self.src}'

class Document:
    def __init__(self, page_content:str, metadata:dict):
        self.page_content = page_content
        self.metadata = metadata
    
    def __repr__(self):
        return f'metadata={self.metadata}, page_content="{self.page_content}"'
    

class DocumentWideMetadata(BaseModel):
    version:str
    docID:str
    timestamp:str
    release:str

class RetrieverResult:
    def __init__(self,firstOrderSpecDocs,secondOrderSpecDocs,retrievedDiscussionDocs):
        self.firstOrderSpecDocs = firstOrderSpecDocs
        self.secondOrderSpecDocs = secondOrderSpecDocs
        self.retrievedDiscussionDocs = retrievedDiscussionDocs

if __name__ == "__main__":
    #print(getFirstPageOfDocxInMarkdown("./data/R299-041.docx"))
    print(getAllFilesInDirMatchingFormat(".",[".py"]))
    print(getDocIDFromText("This is a reference to 3GPP TS 29.041 and also to TS 32.299. Another doc is 24.229. Here is a number 4.3.17"))
