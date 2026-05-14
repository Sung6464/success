from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
import requests
import logging
from pydantic import PrivateAttr
from typing import List, Optional
from dotenv import load_dotenv
import os
from openai import BadRequestError
from langchain_openai import AzureChatOpenAI
from langchain.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()


#Custom LLM class for Langchain using Azure OpenAI
class ContentFilterError(Exception):
    """Raised when Azure OpenAI content management policy is triggered."""
    pass

class AzureCustomLLM(LLM):
    """Custom LLM wrapper for Langchain using Azure OpenAI."""

    _llm: AzureChatOpenAI = PrivateAttr()
    _logger: logging.Logger = PrivateAttr()  # Declare logger as PrivateAttr
    stop: Optional[List[str]] = None

    def __init__(self, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 7000, stream: bool = False, stop: Optional[List[str]] = None):
        super().__init__()
        self._llm = AzureChatOpenAI(
            azure_endpoint = os.getenv("AZURE_OPENAI_LLM_MODEL_API_BASE"),
            api_key = os.getenv("AZURE_OPENAI_LLM_MODEL_API_KEY"),
            azure_deployment= os.getenv("AZURE_OPENAI_LLM_MODEL_LLM_MODEL"),
            api_version = os.getenv("AZURE_OPENAI_LLM_MODEL_API_VERSION"),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
            stop = stop
        )

        # Set up logger (use _logger since it's a PrivateAttr)
        self._logger = logging.getLogger("AzureCustomLLM")
        if not self._logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def _call(self, input: str, stop: Optional[List[str]] = None, sys_prompt: Optional[str] = None, history: Optional[List[str]] = None) -> str:
        messages = [{"role": "system", "content": sys_prompt or "You are a helpful AI assistant."}]
        # if history:
        #     for i in range(0, len(history), 2):
        #         messages.append({"role": "user", "content": history[i]})
        #         if i + 1 < len(history):
        #             messages.append({"role": "assistant", "content": history[i + 1]})
        # messages.append({"role": "user", "content": inputs})

        # formatted_history = []
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content = input))


        try:
            response = self._llm.invoke(messages)
            return response.content
        except BadRequestError as e:
            # Handle Azure OpenAI content filter error
            raise ContentFilterError("Sorry, your request triggered Azure OpenAI's content management policy. Please modify your prompt and try again.") from e
        except ValueError as e:
            # Azure content filter often raises ValueError with a specific message
            if "content filter" in str(e).lower() or "content management policy" in str(e).lower():
                self._logger.error(f"Azure OpenAI content filter triggered: {e}")
                raise ContentFilterError("Sorry, your request triggered Azure OpenAI's content management policy. Please modify your prompt and try again.") from e
            else:
                self._logger.error(f"Azure OpenAI ValueError: {e}")
                raise
        except requests.exceptions.RequestException as e:
            self._logger.error(f"Error during Azure OpenAI API call: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        return "azure_openai_custom_llm"


# llm = AzureCustomLLM()