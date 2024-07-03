import openai
from config import OPENAI_API_KEY

class OpenAIConnection:
    def __init__(self):
        self.api_key = OPENAI_API_KEY

    def generate_prompt_response(self, prompt):
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=150,
            temperature=0.5,
            api_key=self.api_key
        )
        return response.choices[0].text.strip()
