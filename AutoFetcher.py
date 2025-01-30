from settings import config
import requests
from bs4 import BeautifulSoup
# idea:
# These pages are maintained quite consistently. 
# two questions:
# 1) How do we get the latest spec? How do we pass it into the data folder?
# 2) When do we pull and when do we not?

class AutoFetcher:
    def __init__(self,fetch_endpoints):
        self.links = {}
        pass

    def extractLinksFromEndpoint(self,endpoint,params):
        """For a specific endpoint we retrieve the page and collect all the links on that page. 
        Run this for pages which are collections of links.
        @params: search parameters for the request. Try to sort all the links by upload date. 
        The assumption is that the last link after sorting is the most recent one."""
        response = requests.get(endpoint,params=params)
        if response.status_code != 200:
            raise Exception("Error: Could not retrieve page content")
        soup = BeautifulSoup(response.content,'html.parser')
        self.links[endpoint] = []
        for link in soup.find_all('a'):
            self.links[endpoint].append(link.get('href'))
    
    def getMostRecentLink(self,endpoint):
        if not self.links.get(endpoint,[]):
            raise Exception("Error: Run extractLinksFromEndpoint first")
        return self.links[endpoint][-1]

if __name__ == "__main__":
    params = {"sortby":"date"}
    endpoint = "https://www.3gpp.org/ftp/Specs/latest/Rel-16/38_series"
    af = AutoFetcher([endpoint])
    af.extractLinksFromEndpoint(endpoint,params)
    print(f"link is {af.getMostRecentLink(endpoint)}")
