import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig
from data.processors import get_tokenizer, get_image_processor

torch.manual_seed(0)

cfg = VLMConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Change to your own model path after training
path_to_hf_file = hf_hub_download(repo_id="lusxvr/nanoVLM-222M", filename="nanoVLM-222M.pth")
model = VisionLanguageModel(cfg).to(device)
model.load_checkpoint(path_to_hf_file)
model.eval()

tokenizer = get_tokenizer(cfg.lm_tokenizer)
image_processor = get_image_processor(cfg.vit_img_size)

text = "What is this?"
template = f"Question: {text} Answer:"
encoded_batch = tokenizer.batch_encode_plus([template], return_tensors="pt")
tokens = encoded_batch['input_ids'].to(device)

image_path = 'assets/image.png'
image = Image.open(image_path)
image = image_processor(image)
image = image.unsqueeze(0).to(device)

print("Input: ")
print(f'{text}')
print("Output:")
num_generations = 5
for i in range(num_generations):
    gen = model.generate(tokens, image, max_new_tokens=20)
    print(f"Generation {i+1}: {tokenizer.batch_decode(gen, skip_special_tokens=True)[0]}")