import sys
sys.path.append("../")

from abc import ABC,abstractmethod
from openai import OpenAI
from settings import config

class BaseSystemModel(ABC):
    @abstractmethod
    def get_response(self,input):
        pass

class GPTSystemModel(BaseSystemModel):
    def __init__(self):
        self.client = OpenAI(
            api_key=config["API_KEY"],
            timeout=60,
            )
        self.response_template = '''The following is a question about telecommunications and networking. Just give the answer.
        Question:{question}
        Answer:'''
    
    def get_response(self, question):
        response_prompt = self.response_template.format(question=question)

        response = self.client.chat.completions.create(
        model=config["MODEL_NAME"],
        messages=[
                {
                    'role': 'user',
                    'content': response_prompt
                }
            ],
            seed=0,
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    gpt = GPTSystemModel()
    question = "What does the abbreviation \"IAB\" stand for in the context of NR Integrated access and backhaul?"
    print(gpt.get_response(question))