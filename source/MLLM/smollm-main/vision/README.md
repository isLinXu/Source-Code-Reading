
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/SmolVLM.png" width="800" height="auto" alt="Image description">

# SmolVLM
[SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) is a compact open multimodal model that accepts arbitrary sequences of image and text inputs to produce text outputs. It uses SmolLM2-1.7B-Instruct as a language backbone and is designed for efficiency. SmolVLM can answer questions about images, describe visual content, create stories grounded on multiple images, or function as a pure language model without visual inputs. Its lightweight architecture makes it suitable for on-device applications while maintaining strong performance on multimodal tasks.
More details in this blog post: https://huggingface.co/blog/smolvlm

In this section you can find everything related to the training of our Vision Language Models series: SmolVLM. This includes pretraining and finetuning code, as well as evaluation (TODO).

#  Table of Contents
2. [Usage](#usage)
3. [Inference with transformers](#inference-with-transformers)
4. [Inference with mlx-vlm](#inference-with-mlx-vlm)
5. [Video Inference](#video-inference)

## Usage

SmolVLM can be used for inference on multimodal (image + text) tasks where the input comprises text queries along with one or more images. Text and images can be interleaved arbitrarily, enabling tasks like image captioning, visual question answering, and storytelling based on visual content. The model does not support image generation.

To fine-tune SmolVLM on a specific task, you can follow this [fine-tuning tutorial](finetuning/Smol_VLM_FT.ipynb)

## Inference with transformers

You can use transformers to load, infer and fine-tune SmolVLM.

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load images
image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image2 = load_image("https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg")

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "image"},
            {"type": "text", "text": "Can you describe the two images?"}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
"""
Assistant: The first image shows a green statue of the Statue of Liberty standing on a stone pedestal in front of a body of water. 
The statue is holding a torch in its right hand and a tablet in its left hand. The water is calm and there are no boats or other objects visible. 
The sky is clear and there are no clouds. The second image shows a bee on a pink flower. 
The bee is black and yellow and is collecting pollen from the flower. The flower is surrounded by green leaves.
"""
```
## Inference with mlx-vlm

You can also get fast generations for SmolVLM locally with mlx-vlm:
```bash
pip install -U mlx-vlm
python -m mlx_vlm.chat_ui --model mlx-community/SmolVLM-Instruct-8bit
```

## Video inference

Given SmolVLM's long context and the possibility of tweaking the internal frame resizing of the model, we explored its suitability as an accessible option for basic video analysis tasks, particularly when computational resources are limited.

In our evaluation of SmolVLM's video understanding capabilities, we implemented a straightforward video processing pipeline code in [SmolVLM_video_inference.py](../tools/smolvlm_local_inference/SmolVLM_video_inference.py), extracting up to 50 evenly sampled frames from each video while avoiding internal frame resizing. This simple approach yielded surprisingly competitive results on the CinePile benchmark, with a score of 27.14%, a performance that positions the model between InterVL2 (2B) and Video LlaVa (7B).

## Training codebase

The training codebase is available in the [m4](m4) and [experiments](experiments) folders. This codebase is based on an internal codebase from HuggingFace which was in development since 2022. Some of the biggest contributors are:

- [VictorSanh](https://github.com/VictorSanh)
- [HugoLaurencon](https://github.com/HugoLaurencon)
- [SaulLu](https://github.com/SaulLu)
- [leot13](https://github.com/leot13)
- [stas00](https://github.com/stas00)
- [apsdehal](https://github.com/apsdehal)
- [thomasw21](https://github.com/thomasw21)
- [siddk](https://github.com/siddk)

