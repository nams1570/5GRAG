from difflib import Differ
from MetadataAwareChunker import getFullSectionChunks
from langchain_core.documents import Document

SENTENCE_SEPARATOR = ". "

def get_all_section_names(chunks:list[Document])->set:
    all_section_names = set()
    for chunk in chunks:
        all_section_names.add(chunk.metadata["section"])
    return all_section_names

class ChangeTracker:

    @staticmethod
    def areAdjacentVersions(fromVersion,toVersion):
        x1, y1, z1 = map(int, fromVersion.split('.'))
        x2, y2, z2 = map(int, toVersion.split('.'))
        
        # Check if x values are equal
        if x1 != x2:
            return False

        # Check for adjacency in y (and z must be same)
        if y2 - y1 == 1:
            return True

        # Check for adjacency in z (and y must be same)
        if z2 - z1 == 1 and y1 == y2:
            return True

        return False

    @staticmethod
    def getChanges(chunk1:Document,chunk2:Document)->dict: 
        """@chunk1 and @chunk2 are two Documents representing the same chunk across different versions.
        Assumption is that chunk2 is the later version.
        Returns: change object dictionary. Has keys 'from_version', 'to_version', 'changes'.
        """
        if "version" not in chunk1.metadata or "version" not in chunk2.metadata:
            raise Exception("Cannot diff as no version metadata in one or both of the chunks")
        
        if chunk1.metadata["docID"] != chunk2.metadata["docID"]:
            raise Exception("Not the same release")
        
        text1,text2 = chunk1.page_content, chunk2.page_content
        from_version,to_version = chunk1.metadata["version"], chunk2.metadata["version"]

        change_metadata = {'from_version':from_version,'to_version':to_version,'docID':chunk1.metadata["docID"],'section':chunk2.metadata["section"]}

        change_obj = {'metadata':change_metadata,'changes':{'add':[],'remove':[]}}
        for delta in Differ().compare(text1.split(SENTENCE_SEPARATOR),text2.split(SENTENCE_SEPARATOR)):
            if len(delta)>0 and delta[0] == "+":
                change_obj['changes']['add'].append(delta[1:])
            elif len(delta) >0 and delta[0] == '-':
                change_obj['changes']['remove'].append(delta[1:])

        return change_obj
    
    @staticmethod
    def convertChangeObjToDocument(change_obj:dict)->list[Document]:
        doc_list = []
        for change in change_obj['changes']['add']:
            doc_list.append(Document(metadata=change_obj["metadata"],page_content=change))
        return doc_list
    
    @staticmethod
    def createDBDocumentsForAdjacentVersions(fromVersionChunks:list[Document],toVersionChunks:list[Document])->list[Document]:
        """Here, we take in two chunk lists of versions that must be adjacent and for each section, generate a list of Documents representing the new content added"""
        if fromVersionChunks == [] or toVersionChunks == []:
            raise Exception("Error: cannot diff empty chunklist")
        if not ChangeTracker.areAdjacentVersions(fromVersionChunks[0].metadata["version"],toVersionChunks[0].metadata["version"]):
            raise Exception("Error: must be passed versions that are adjacent")

        fromSectionMap,toSectionMap = {},{}

        for doc in fromVersionChunks:
            if doc.metadata["section"] in fromSectionMap:
                raise Exception("Error (fromVersion): must be passed section wide chunks. Cannot have multiple chunks per section")
            fromSectionMap[doc.metadata["section"]] = doc
        
        for doc in toVersionChunks:
            if doc.metadata["section"] in toSectionMap:
                raise Exception("Error (toVersion): must be passed section wide chunks. Cannot have multiple chunks per section")
            toSectionMap[doc.metadata["section"]] = doc

        DBDocList = []

        for section_name in toSectionMap.keys():
            change_obj = ChangeTracker.getChanges(fromSectionMap[section_name],toSectionMap[section_name])
            DBDocList.extend(ChangeTracker.convertChangeObjToDocument(change_obj))
        
        return DBDocList

if __name__ == "__main__":
    file_list = ["./testchange/38211-i60.docx","./testchange/38211-i70.docx"]
    """section = "6.3.3.1"

    sectioned_things = {}
    for doc in getFullSectionChunks([file_list[0]]):
        sectioned_things[doc.metadata["section"]] = sectioned_things.get(doc.metadata["section"],[]) + [doc]

    for doc in getFullSectionChunks([file_list[1]]):
        sectioned_things[doc.metadata["section"]] = sectioned_things.get(doc.metadata["section"],[]) + [doc]
    
    change_obj = ChangeTracker.getChanges(sectioned_things[section][0],sectioned_things[section][1])
    print(ChangeTracker.convertChangeObjToDocument(change_obj))"""
    fromVersionChunks = getFullSectionChunks([file_list[0]])
    toVersionChunks = getFullSectionChunks([file_list[1]])
    print(ChangeTracker.createDBDocumentsForAdjacentVersions(fromVersionChunks,toVersionChunks))
    