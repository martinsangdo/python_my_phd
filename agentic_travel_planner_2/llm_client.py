import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

class GeminiLLM:
    def __init__(self, model_name="models/gemini-pro"):
        self.model = genai.GenerativeModel(model_name)

    def call(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text