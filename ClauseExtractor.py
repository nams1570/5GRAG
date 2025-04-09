from docx import Document

FILENAME = "./data/38214-hc0.docx"

f = open(FILENAME,'rb')
doc = Document(f)

def iter_readings(paragraphs):
    for paragraph in paragraphs:
        if paragraph.style.name.startswith('Heading'):
            yield paragraph

def iter_paragraphs(doc):
    for paragraph in doc.paragraphs:
        yield paragraph.text

def iter_sections(doc):
    for part in doc.iter_inner_content():
        if part.style.name.startswith('Heading'):
            yield "hd: "+part.text
        elif "table" in part.style.name.lower():
            yield "tbl: " + part.style.name
        else:
            yield part.text


if __name__ == "__main__":
    for heading in iter_sections(doc):
        print(heading)
