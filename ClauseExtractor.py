from docx import Document

FILENAME = "./data/38214-hc0.docx"

f = open(FILENAME,'rb')
doc = Document(f)

p = doc.add_paragraph('This is a paragraph ')

def iter_readings(paragraphs):
    for paragraph in paragraphs:
        if paragraph.style.name.startswith('Heading'):
            yield paragraph


if __name__ == "__main__":
    for heading in iter_readings(doc.paragraphs):
        print(heading.text)
