import re

class ReferenceExtractor:
    def __init__(self):
        self.regxs=[]
        patterns = [r"(clause\s+(\d+(.\d+)*).?(of \[\d+, [A-Za-z0-9_\. ]*\])?)",r"(Table\s+(\d+([.\d+|\-\d])*).?)"]
        for pattern in patterns:
            #re.compile turns a string into a regex. 
            self.regxs.append(re.compile(pattern,re.IGNORECASE))

    def findAllMatches(self,doc:str):
        """@doc: the string that we compare against all regexes
        returns: a list of the substrings that match any of the regexes"""
        matchedStrings = []
        for regx in self.regxs:
            temp = regx.findall(doc)
            if temp:
                for tup in temp:
                    matchedStrings.append(tup[0]) 
        return matchedStrings

if __name__ == "__main__":
    examples = ["The determination of the used resource allocation table is defined in clause 6.1.2.1.1 of [4, TS 38.211] though you can also check Clause 6.2 or clause 6.3.",
    "Aperiodic CSI-RS is configured and triggered/activated as described in Clause 8.5.1.2", 
    "The UE shall derive CQI as specified in clause 5.2.2.1", "the UE procedure for receiving the PDSCH upon detection of a PDCCH follows clause 5.1 and the QCL assumption for the PDSCH as defined in clause 5.1.5"]
    re = ReferenceExtractor()
    print(f"{re.findAllMatches(examples[3])}")