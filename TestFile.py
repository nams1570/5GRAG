from docx import Document as DocParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

FILENAME = "./data/38214-hc0.docx"

f = open(FILENAME,'rb')
doc = DocParser(f)

def getSectionedChunks(doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

    current_section = None
    sections = []
    chunks_with_metadata= []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if paragraph.style.name.startswith('Heading'):
            current_section = paragraph.text
        elif text:
            sections.append((text,current_section))
        
    for text,section in sections:
        split_chunks = text_splitter.split_text(text)
        for chunk in split_chunks:
            chunks_with_metadata.append({
                "page_content":chunk,
                "metadata":{'section':section}
            })
    return chunks_with_metadata

if __name__ =="__main__":
    print(getSectionedChunks(doc))