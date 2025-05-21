from .base import SmolTool
from typing import Generator, Optional
from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class SummaryMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime

class SmolSummarizer(SmolTool):
    def __init__(self):
        self.name = "SmolLM2-1.7B"
        
        super().__init__(
            model_repo="andito/SmolLM2-1.7B-Instruct-F16-GGUF",
            model_filename="smollm2-1.7b-8k-dpo-f16.gguf",
            system_prompt="Concisely summarize the main points of the input text in up to three sentences, focusing on key information and events.",
        )

    def process(self, text: str, question: Optional[str] = None) -> Generator[str, None, None]:
        if question is None:
            print("Summarizing text")
            prompt = f"{self.prefix_text}\n{text}"
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "This is a short summary of the text:"}
            ]
        else:
            print("Answering question")
            prompt = f"Original text:\n{text}\n\nQuestion: {question}"
            messages = [
                {"role": "user", "content": prompt},
            ]

        for chunk in self._create_chat_completion(messages, max_tokens=1024, temperature=0.1, top_p=0.9):
            yield chunk
