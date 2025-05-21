from .base import SmolTool
from typing import Generator

class SmolRewriter(SmolTool):
    def __init__(self):
        super().__init__(
            model_repo="andito/SmolLM2-1.7B-Instruct-F16-GGUF",
            model_filename="smollm2-1.7b-8k-dpo-f16.gguf",
            system_prompt="You are an AI writing assistant. Your task is to rewrite the user's email to make it more professional and approachable while maintaining its main points and key message. Do not return any text other than the rewritten message.",
            prefix_text="Rewrite the message below to make it more professional and approachable while maintaining its main points and key message. Do not add any new information or return any text other than the rewritten message\nThe message:"
        )

    def process(self, text: str) -> Generator[str, None, None]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{self.prefix_text}\n{text}"}
        ]
        yield from self._create_chat_completion(messages, temperature=0.4, repeat_penalty=1.0, top_k=0, max_tokens=1024)