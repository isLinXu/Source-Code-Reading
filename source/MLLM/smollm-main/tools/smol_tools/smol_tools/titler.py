from .base import SmolTool
from typing import Generator

class SmolTitler(SmolTool):
    def __init__(self):
        super().__init__(
            model_repo="andito/SmolLM2-1.7B-Instruct-F16-GGUF",
            model_filename="smollm2-1.7b-8k-dpo-f16.gguf",
            system_prompt="",
            prefix_text="Create a title for this conversation:",
        )

    def process(self, text: str) -> Generator[str, None, None]:
        messages = [
            {"role": "user", "content": f"{self.prefix_text}\n{text}"}
        ]
        yield from self._create_chat_completion(messages, max_tokens=128, temperature=0.6, top_p=0.9, top_k=0, repeat_penalty=1.1)