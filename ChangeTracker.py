from difflib import Differ
from MetadataAwareChunker import getFullSectionChunks,addExtraDocumentWideMetadataForReason
from utils import Document

SENTENCE_SEPARATOR = ". "

def get_doc_list_for_version_preceding_first(metadata:dict,version:str):
    metadata = {**metadata,'version':version,'timestamp':'1800-01'}
    return [Document(metadata=metadata,page_content='...')]

def get_empty_document(metadata:dict):
    return Document(metadata=metadata,page_content='')


def get_all_section_names(chunks:list[Document])->set:
    all_section_names = set()
    for chunk in chunks:
        all_section_names.add(chunk.metadata["section"])
    return all_section_names

def get_version_preceding_first_in_release(releaseNum)->str:
    """Assumption is that the first version of a release is x.0.0"""
    return f"{releaseNum}" + ".0.-1"

class ChangeTracker:

    @staticmethod
    def areAdjacentVersions(fromVersion,toVersion):
        x1, y1, z1 = map(int, fromVersion.split('.'))
        x2, y2, z2 = map(int, toVersion.split('.'))

        if z1 == -1:
            if y2 == 0 and z2 == 0:
                return True
            return False
        
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

        change_metadata = {'from_version':from_version,'to_version':to_version,'docID':chunk1.metadata["docID"],'section':chunk2.metadata["section"],'fromTimestamp':chunk1.metadata["timestamp"],'toTimestamp':chunk2.metadata["timestamp"]}

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
                return []
                raise Exception(f"Error (fromVersion): must be passed section wide chunks. Cannot have multiple chunks per section. Version is {doc.metadata['version']}, docID is {doc.metadata['docID']}, for section {doc.metadata['section']}")
            fromSectionMap[doc.metadata["section"]] = doc
        
        for doc in toVersionChunks:
            if doc.metadata["section"] in toSectionMap:
                return []
                raise Exception(f"Error (toVersion): must be passed section wide chunks. Cannot have multiple chunks per section. Version is {doc.metadata['version']}, docID is {doc.metadata['docID']}, for section {doc.metadata['section']}")
            toSectionMap[doc.metadata["section"]] = doc

        DBDocList = []

        for section_name in fromSectionMap.keys():
            if fromSectionMap[section_name]:
                def_metadata = fromSectionMap[section_name].metadata

        for section_name in toSectionMap.keys():
            change_obj = ChangeTracker.getChanges(fromSectionMap.get(section_name,get_empty_document({**def_metadata,'section':section_name})),toSectionMap[section_name])
            DBDocList.extend(ChangeTracker.convertChangeObjToDocument(change_obj))
        
        return DBDocList

if __name__ == "__main__":
    file_list = ["./testchange/38211-f10.docx","./testchange/38211-f00.docx"]
    """section = "6.3.3.1"

    sectioned_things = {}
    for doc in getFullSectionChunks([file_list[0]]):
        sectioned_things[doc.metadata["section"]] = sectioned_things.get(doc.metadata["section"],[]) + [doc]

    for doc in getFullSectionChunks([file_list[1]]):
        sectioned_things[doc.metadata["section"]] = sectioned_things.get(doc.metadata["section"],[]) + [doc]
    
    change_obj = ChangeTracker.getChanges(sectioned_things[section][0],sectioned_things[section][1])
    print(ChangeTracker.convertChangeObjToDocument(change_obj))"""
    fromVersionChunks = getFullSectionChunks([file_list[0]])
    toVersionChunks = getFullSectionChunks([file_list[1]],addExtraDocumentWideMetadataForReason)
    #print(ChangeTracker.createDBDocumentsForAdjacentVersions(fromVersionChunks,toVersionChunks))
    for chunk in toVersionChunks:
        if chunk.metadata['section'] == '6.3.1.1':
            print(chunk)
            fake_metadata = {**chunk.metadata,'version':'15.0-1','timestamp':'1800-01'}
            fake_chunk = get_empty_document(fake_metadata)
            print(ChangeTracker.getChanges(fake_chunk,chunk))
            exit(0)
    