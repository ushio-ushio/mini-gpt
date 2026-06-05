from typing import Any, List, Optional
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import Generation, LLMResult
from pydantic import Field
import requests

class LocalDecoderLLM(BaseLLM):
    api_url: str = Field(default="http://127.0.0.1:8000/generate")

    @property
    def _llm_type(self) -> str:
        return "custom_local_decoder"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            payload = {
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 128)
            }
            try:
                response = requests.post(self.api_url, json=payload, timeout=60)
                response.raise_for_status()
                result_text = response.json().get("response", "")
                generations.append([Generation(text=result_text)])
            except requests.exceptions.RequestException as e:
                error_msg = f"API request failed: {str(e)}"
                generations.append([Generation(text=error_msg)])
                
        return LLMResult(generations=generations)