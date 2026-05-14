from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
import requests
from typing import List, Optional
from dotenv import load_dotenv
import os

load_dotenv()


class CustomLLM(LLM):
    """Custom LLM wrapper for LangChain using a REST API."""
    model:str = os.getenv('QUASAR_MODEL')
    endpoint_url: str = os.getenv('QUASAR_ENDPOINT_URL')
    headers: dict = {
        "Content-Type": "application/json", 
        "X-API-KEY": os.getenv('QUSAR_API_KEY')
    }
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4000
    # stream: bool = False
    stop: Optional[List[str]] = None
    

    def _call(self,input:str, stop: Optional[List[str]] = None,sys_prompt:Optional[str] = None , history:Optional[list[str]|None]= None) -> str:
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        if history:
            messages = messages+history
        messages.append({"role": "user", "content": input})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if stop:
            payload["stop"] = stop

        response = requests.post(self.endpoint_url, headers=self.headers, json=payload)
        response.raise_for_status()
        data = response.json()

        return data['choices'][0]['message']['content']

    @property
    def _llm_type(self) -> str:
        return "custom-llm"

