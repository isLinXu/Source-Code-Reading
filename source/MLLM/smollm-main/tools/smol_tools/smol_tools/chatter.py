from .base import SmolTool
from typing import Generator, List, Dict
from dataclasses import dataclass
from datetime import datetime
import json
import os

@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime

    # Add methods to convert to/from dict for JSON serialization
    def to_dict(self):
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

class SmolChatter(SmolTool):
    def __init__(self):
        self.chat_history: List[ChatMessage] = []
        self.chat_archive: Dict[str, List[ChatMessage]] = {}
        self.current_chat_id = None
        self.chats_dir = "saved_chats"
        self._original_chat_state = None  # To track modifications
        self.name = "SmolLM2-1.7B"
        
        # Create chats directory if it doesn't exist
        if not os.path.exists(self.chats_dir):
            os.makedirs(self.chats_dir)
            
        super().__init__(
            model_repo="andito/SmolLM2-1.7B-Instruct-F16-GGUF",
            model_filename="smollm2-1.7b-8k-dpo-f16.gguf",
            system_prompt="You are a helpful AI assistant named SmolLM, trained by Hugging Face..",
        )

    def start_new_chat(self):
        """Start a new chat with a unique ID"""
        self.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.chat_history = []
        self._original_chat_state = None

    def has_current_chat(self) -> bool:
        """Check if there are any messages in the current chat"""
        return len(self.chat_history) > 0

    def save_current_chat(self, title: str = None, overwrite: bool = False):
        """Save the current chat to disk if it has any messages"""
        if not self.chat_history:
            return
            
        if title:
            # If overwriting, use existing chat_id if it matches the title
            if not overwrite or self.current_chat_id != title:
                self.current_chat_id = title
        elif not self.current_chat_id:
            self.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Convert chat history to serializable format
        chat_data = {
            'id': self.current_chat_id,
            'messages': [msg.to_dict() for msg in self.chat_history]
        }
        
        # Save to file
        filename = f"{self.chats_dir}/chat_{self.current_chat_id}.json"
        with open(filename, 'w') as f:
            json.dump(chat_data, f)
            
        # Update original state to reflect saved state
        self._original_chat_state = [msg.to_dict() for msg in self.chat_history]

    def load_chat(self, chat_id: str):
        """Load a specific chat from disk"""
        filename = f"{self.chats_dir}/chat_{chat_id}.json"
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.current_chat_id = data['id']
                self.chat_history = [ChatMessage.from_dict(msg) for msg in data['messages']]
                # Store original state for modification tracking
                self._original_chat_state = [msg.to_dict() for msg in self.chat_history]
        except FileNotFoundError:
            print(f"Chat {chat_id} not found")

    def is_chat_modified(self) -> bool:
        """Check if the current chat has been modified since loading"""
        if self._original_chat_state is None:
            # New chat that hasn't been saved yet
            return len(self.chat_history) > 0
            
        current_state = [msg.to_dict() for msg in self.chat_history]
        return current_state != self._original_chat_state

    def get_saved_chats(self) -> List[str]:
        """Get list of saved chat IDs"""
        chats = []
        for filename in os.listdir(self.chats_dir):
            if filename.startswith('chat_') and filename.endswith('.json'):
                chat_id = filename[5:-5]  # Remove 'chat_' prefix and '.json' suffix
                chats.append(chat_id)
        return sorted(chats, reverse=True)  # Most recent first

    def _warm_up(self):
        super()._warm_up()
        self.clear_chat_history()

    def process(self, text: str) -> Generator[str, None, None]:
        # Add user message to history
        self.chat_history.append(ChatMessage(
            role="user",
            content=text,
            timestamp=datetime.now()
        ))
        
        # Build messages including chat history
        messages = [{"role": "system", "content": self.system_prompt}]
        # Include last 5 messages for context
        for msg in self.chat_history:
            messages.append({"role": msg.role, "content": msg.content})

        # Generate response
        response = ""
        for chunk in self._create_chat_completion(messages, max_tokens=1024):
            response = chunk
            yield chunk
        
        # Add assistant's response to history
        self.chat_history.append(ChatMessage(
            role="assistant",
            content=response,
            timestamp=datetime.now()
        ))

    def get_chat_history(self) -> List[ChatMessage]:
        return self.chat_history
    
    def clear_chat_history(self):
        self.chat_history = []

    def get_current_chat_id(self) -> str:
        """Get the ID of the current chat"""
        return self.current_chat_id