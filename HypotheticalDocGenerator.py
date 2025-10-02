from settings import config
from openai import OpenAI
import time

MAX_RETRIES = 3
DELAY = 2

class HypotheticalDocGenerator:
    def __init__(self, api_key=config["API_KEY"], model_name=config["MODEL_NAME"]):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def generate_hypothetical_document(self, query: str) -> str:
        """Generates a hypothetical document based on the input query. 
        Sometimes questions dont contain much detail and so are not useful for retrieval.
        This function uses an LLM to generate a hypothetical document that contains more detail."""

        system_prompt = {
            "role": "system",
            "content":"Write a passage to answer the question in about 200 to 300 words."
        }
        user_prompt= {
            "role": "user",
            "content": query
        }

        for attempt in range(1,MAX_RETRIES+1,1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages = [system_prompt, user_prompt],
                )
            except Exception as e:
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(DELAY)
                else:
                    print(f"Max retries reached, returning None.")
                    return None
        return response.choices[0].message.content.strip()

if __name__ == "__main__":
    hdg = HypotheticalDocGenerator()
    query = "How many maximum 5QI we can create under one PDU Session?"
    print(hdg.generate_hypothetical_document(query))

    