import sys
sys.path.append("../")
from ReferenceExtractor import ReferenceExtractor
from MetadataAwareChunker import getFullSectionChunks

RE = ReferenceExtractor()

class GraphNode:
    def __init__(self,section_name:str,section_text:str,metadata:dict):
        """@section_name: the heading or subheading of the section
        @section_text: should be the entire section text. It will be scanned for references"""
        self.section_name = section_name
        self.section_text = section_text
        self.metadata = metadata
        self.neighbors = self.getNeighborsFromText(section_text)
    
    def getNeighborsFromText(self,section_text:str)->list[str]:
        refObjs = RE.runREWithStrList([section_text])
        intraDocNeighbors = RE.extractClauseNumbersOfSrc(refObjs)
        return intraDocNeighbors
    
    def toJSON(self):
        return {
            "section_name":self.section_name,
            "section_text":self.section_text,
            "metadata":self.metadata,
            "neighbors":self.neighbors,
        }
    
if __name__ == "__main__":
    file_list = ["../data/38211-i60.docx"]
    section_chunks = getFullSectionChunks(file_list)
    for test_chunk in section_chunks:
        section_name,section_text,metadata = test_chunk.metadata["section"],test_chunk.page_content,test_chunk.metadata
        gn = GraphNode(section_name,section_text,metadata)
        if gn.neighbors:
            print(gn.toJSON())