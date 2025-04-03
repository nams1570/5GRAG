from docx import Document as DocParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def getSectionedChunks(file_list):
    """@input: file_list. List of files in relative path that will be chunked.
    Returns: master list chunks_with_metadata that has chunks of all the files stored as langchain Documents.
    These Documents have section metadata"""
    chunks_with_metadata= []
    for file in file_list:

        f = open(file,'rb')
        doc = DocParser(f)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

        current_section = None
        sections = []

        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if paragraph.style.name.startswith('Heading'):
                current_section = paragraph.text
            elif text:
                sections.append((text,current_section))
            
        for text,section in sections:
            split_chunks = text_splitter.split_text(text)
            for chunk in split_chunks:
                chunks_with_metadata.append(Document(
                    page_content=chunk,
                    metadata={'source':file,'section':section}
                ))
    return chunks_with_metadata

if __name__ =="__main__":
    file_list = ["./data/38214-hc0.docx"]
    print(getSectionedChunks(file_list)[-100])