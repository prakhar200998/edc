import openai

class OpenAIConnection:
    def __init__(self, api_key):
        self.api_key = api_key

    def generate_prompt_response(self, prompt):
        response = openai.Completion.create(
            engine="gpt-4.0-turbo",
            prompt=prompt,
            max_tokens=150,
            temperature=0.5,
            api_key=self.api_key
        )
        return response.choices[0].text.strip()
