from openai import OpenAI
from settings import config
from utils import Document

class RAGQAEngine:
    def __init__(self, prompt_template_file_path:str, model_name:str=config["MODEL_NAME"],api_key:str=config["API_KEY"]):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

        with open(prompt_template_file_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
    
    def build_context(self,docs: list[Document])->list[str]:
        return [f"{{metadata: {doc.metadata},\n content: {doc.page_content} }}" for doc in docs]

    def build_prompt(self,query:str,docs:list[Document])->str:
        formatted_docs = self.build_context(docs)
        return self.prompt_template.format(input=query,context=formatted_docs)
    
    def get_answer(self,query:str, docs:list[Document]) -> str:
        prompt = self.build_prompt(query,docs)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    
if __name__ == "__main__":
    e = RAGQAEngine(prompt_template_file_path="prompt.txt")
    docs = [Document(page_content="5G is the fifth generation mobile network.", metadata={"source": "spec1"}),
            Document(page_content="It offers faster speeds and more reliable connections.", metadata={"source": "spec2"})]
    answer = e.get_answer("What is 5G?", docs)
    print(answer)
