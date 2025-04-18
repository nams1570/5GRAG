from docx import Document as DocParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def parse_table(table):
    """@input: docx table. Consists of rows and cells and the cells may have other tables.
    This is a helper function to help parse tables from docx. 
    It is needed as tables may contain nested tables."""
    for row_idx,row in enumerate(table.rows):
        for col_idx,col in enumerate(row.cells):
            if col.tables:
                for nested_table in col.tables:
                    parse_table(nested_table)
            else:
                yield col.text

def process_section_name(section_name:str)->str:
    """@input: (str)  section name.
    This is a helper function to process the section name into a numerical format.
    We do this to keep it consistent with how external references will be searched for.
    """
    return section_name.split("\t")[0]

def getSectionedChunks(file_list):
    """@input: file_list. List of files in relative path that will be chunked.
    Returns: master list chunks_with_metadata that has chunks of all the files stored as langchain Documents.
    These Documents have section metadata"""
    chunks_with_metadata= []
    for file in file_list:

        f = open(file,'rb')
        doc = DocParser(f)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,add_start_index=True)

        current_section_title = "N/A"
        current_section_text = ""
        sections = []
        #TODO: Switch to collecting a "current section" list of text chunks
        # currently there's no overlap between paragraphs in the same section which is bad for retrieval
        # We want to collect (section_text,current_section) where section_text is a combination of paragraphs
        # Then, we use the text_splitter to split it into split chunks
        for part in doc.iter_inner_content():
            if part.style.name.startswith('Heading'):
                # update current section
                sections.append((current_section_text,current_section_title))
                current_section_text = ""
                current_section_title = part.text
            elif "table" in part.style.name.lower():
                for table_datum in parse_table(part):
                    current_section_text += table_datum
            else:
                # append text to current section
                current_section_text += part.text
        
        #Add last section to sections
        sections.append((current_section_text,current_section_title))

        for text,section_name in sections:
            split_chunks = text_splitter.split_text(text)
            for chunk in split_chunks:
                chunks_with_metadata.append(Document(
                    page_content=chunk,
                    metadata={'source':file,'section':process_section_name(section_name)}
                ))
    return chunks_with_metadata

def getFullFileChunks(file_list):
    chunks = []
    for file in file_list:

        f = open(file,'rb')
        doc = DocParser(f)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,add_start_index=True)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text
        split_chunks = text_splitter.split_text(text)
        for chunk in split_chunks:
            chunks.append(Document(
                page_content=chunk,
                metadata={'source':file}
            ))
    return chunks

if __name__ =="__main__":
    file_list = ["./data/38101-5-hb0.docx"]
    print(getSectionedChunks(file_list)[0:40])