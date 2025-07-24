from zipfile import ZipFile
import os
import subprocess
from markitdown import MarkItDown
from openai import OpenAI
from pydantic import BaseModel
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


##############################
### Chunking Tools Section ###
##############################

def getFirstPageOfDocxInMarkdown(filepath:str):
    """filepath must be an absolute path"""
    md = MarkItDown(enable_plugins=False) # Set to True to enable plugins
    result = md.convert(filepath)
    return result.text_content[:500]

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
                        2. docID can be extracted by looking for something of the form 'az x.y' where x and y are numbers and az represents any two letters. This usually follows the phrase 3GPP \
                            3. the timestamp is a date of the form YYYY-MM\
                                4. Release is a number usually prefaced by the word 'Release' \
                                    5. If you cannot instantiate a field, populate it with the empty string."
            },
            {"role":"user","content":text_chunk},
        ],
        text_format=DocumentWideMetadata 
    )
    return dict(response.output_parsed)

######################
## Tokenizing Tools ##
######################

def getTokenCount(text:str,model_name:str,supressWarning:bool=True):
    """Use this to get a picture of how many tokens the @text contains."""
    if "gpt" in model_name:
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
    

class DocumentWideMetadata(BaseModel):
    version:str
    docID:str
    timestamp:str
    release:str

class RetrieverResult:
    def __init__(self,firstOrderSpecDocs,secondOrderSpecDocs):
        self.firstOrderSpecDocs = firstOrderSpecDocs
        self.secondOrderSpecDocs = secondOrderSpecDocs

if __name__ == "__main__":
    #print(getFirstPageOfDocxInMarkdown("./data/R299-041.docx"))
    print(getAllFilesInDirMatchingFormat(".",[".py"]))
