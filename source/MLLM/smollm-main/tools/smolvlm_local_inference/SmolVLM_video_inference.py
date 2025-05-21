import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from PIL import Image
import cv2
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoFrameExtractor:
    def __init__(self, max_frames: int = 50):
        self.max_frames = max_frames
        
    def resize_and_center_crop(self, image: Image.Image, target_size: int) -> Image.Image:
        # Get current dimensions
        width, height = image.size
        
        # Calculate new dimensions keeping aspect ratio
        if width < height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))
            
        # Resize
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center crop
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        
        return image.crop((left, top, right, bottom))
        
    def extract_frames(self, video_path: str) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Calculate frame indices to extract (1fps)
        frame_indices = list(range(0, total_frames, fps))
        
        # If we have more frames than max_frames, sample evenly
        if len(frame_indices) > self.max_frames:
            indices = np.linspace(0, len(frame_indices) - 1, self.max_frames, dtype=int)
            frame_indices = [frame_indices[i] for i in indices]
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame)
                pil_image = self.resize_and_center_crop(pil_image, 384)
                frames.append(pil_image)
        
        cap.release()
        return frames

def load_model(checkpoint_path: str, base_model_id: str = "HuggingFaceTB/SmolVLM-Instruct", device: str = "cuda"):
    # Load processor from original model
    processor = AutoProcessor.from_pretrained(base_model_id)
    if checkpoint_path:
        # Load fine-tuned model from checkpoint
        model = Idefics3ForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
    else:
        model = Idefics3ForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map=device
        )    

    # Configure processor for video frames
    processor.image_processor.size = (384, 384)
    processor.image_processor.do_resize = False
    processor.image_processor.do_image_splitting = False
    
    return model, processor

def generate_response(model, processor, video_path: str, question: str, max_frames: int = 50):
    # Extract frames
    frame_extractor = VideoFrameExtractor(max_frames)
    frames = frame_extractor.extract_frames(video_path)
    logger.info(f"Extracted {len(frames)} frames from video")
    
    # Create prompt with frames
    image_tokens = [{"type": "image"} for _ in range(len(frames))]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Answer briefly."},
                *image_tokens,
                {"type": "text", "text": question}
            ]
        }
    ]

    # Process inputs
    inputs = processor(
        text=processor.apply_chat_template(messages, add_generation_prompt=True),
        images=[img for img in frames],
        return_tensors="pt"
    ).to(model.device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=5,
        temperature=0.7,
        do_sample=True,
        use_cache=True
    )
    
    # Decode response
    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Configuration
    #checkpoint_path = "/path/to/your/checkpoint"
    checkpoint_path = None
    base_model_id = "HuggingFaceTB/SmolVLM-Instruct"  
    video_path = "/path/to/video.mp4"
    question = "Describe the video"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    logger.info("Loading model...")
    model, processor = load_model(checkpoint_path, base_model_id, device)
    
    # Generate response
    logger.info("Generating response...")
    response = generate_response(model, processor, video_path, question)
    
    # Print results
    print("Question:", question)
    print("Response:", response)

if __name__ == "__main__":
    main()