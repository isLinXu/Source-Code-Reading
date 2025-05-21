from abc import ABC, abstractmethod
from typing import Generator, List, Dict, Any, Union, Tuple
from llama_cpp import Llama

class SmolTool(ABC):
    # Class-level cache for model instances
    _model_cache: Dict[Tuple[str, str], Llama] = {}

    def __init__(self, model_repo: str, model_filename: str, system_prompt: str, prefix_text: str = "", n_ctx: int = 8192):
        self.system_prompt = system_prompt
        self.prefix_text = prefix_text
        
        # Create a cache key from the model repo and filename
        cache_key = (model_repo, model_filename)
        
        # Track if this is a new model load
        is_new_model = cache_key not in self._model_cache
        
        # Try to get the model from cache, or create and cache a new one
        if is_new_model:
            print(f"Loading model {model_filename} from {model_repo}...")
            self._model_cache[cache_key] = Llama.from_pretrained(
                repo_id=model_repo,
                filename=model_filename,
                n_ctx=n_ctx,
                verbose=False
            )
        
        self.model = self._model_cache[cache_key]
        
        # Only warm up for newly loaded models
        if is_new_model:
            self._warm_up()

    def _warm_up(self):
        """Warm up the model with a test prompt"""
        print(f"Warming up {self.__class__.__name__}...")
        test_text = "This is a test message to warm up the model."
        # Consume the generator to complete the warm-up
        for _ in self.process(test_text):
            pass
        print(f"{self.__class__.__name__} ready!")

    @abstractmethod
    def process(self, text: str) -> Generator[str, None, None]:
        """Process the input text and yield results as they're generated"""
        pass

    def _create_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.4,
        top_p: float = 0.9,
        top_k: int = 50,
        repeat_penalty: float = 1.2,
        max_tokens: int = 256
    ) -> Generator[str, None, None]:
        """Helper method to create chat completions with standard parameters"""
        output = ""
        for chunk in self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stream=True
        ):
            content = chunk['choices'][0]['delta'].get('content')
            if content:
                if content in ["<end_action>", "<|endoftext|>"]:
                    break
                output += content
                yield output