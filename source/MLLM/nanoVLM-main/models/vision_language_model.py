from models.vision_transformer import ViT  # 从本地模块导入Vision Transformer (ViT)
from models.language_model import LanguageModel  # 从本地模块导入Language Model
from models.modality_projector import ModalityProjector  # 从本地模块导入Modality Projector

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数模块

class VisionLanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()  # 调用父类构造函数
        self.cfg = cfg  # 保存配置对象
        self.vision_encoder = ViT(cfg)  # 初始化视觉编码器 (ViT)
        self.decoder = LanguageModel(cfg)  # 初始化语言模型 (Decoder)
        self.MP = ModalityProjector(cfg)  # 初始化模态投影器

    def forward(self, input_ids, image, attention_mask=None, targets=None):
        image_embd = self.vision_encoder(image)  # 通过视觉编码器处理图像，获取图像嵌入
        image_embd = self.MP(image_embd)  # 通过模态投影器将图像嵌入投影到语言模型的维度空间

        token_embd = self.decoder.token_embedding(input_ids)  # 对输入的token ID进行嵌入

        combined_embd = torch.cat((image_embd, token_embd), dim=1) # Concatenate image embeddings to token embeddings # 将图像嵌入和token嵌入沿序列长度维度拼接起来
        
        # Adjust attention mask to account for image tokens
        # 调整注意力掩码以考虑图像token
        if attention_mask is not None:
            # Create mask of 1s for image tokens (all image tokens should be attended to)
            # 为图像token创建全1的掩码（所有图像token都应该被关注）
            batch_size = image_embd.size(0)  # 获取批次大小
            img_seq_len = image_embd.size(1)  # 获取图像序列长度
            image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)  # 创建图像注意力掩码
            
            # Combine image and token attention masks
            # 合并图像和token的注意力掩码
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)  # 将图像掩码和token掩码拼接起来

        logits = self.decoder(combined_embd, attention_mask) # Not logits yet, but easier to return like this # 通过语言模型解码器处理合并后的嵌入和注意力掩码（此时输出可能不是最终logits，取决于decoder配置）

        loss = None  # 初始化损失为None
        if targets is not None:  # 如果提供了目标（用于训练）
            # Only use the token part of the logits for loss computation
            # 只使用输出中对应token部分的logits来计算损失
            logits = self.decoder.head(logits)  # 如果decoder输出的是嵌入，通过head层转换为logits
            logits = logits[:, image_embd.size(1):, :]  # 截取输出中对应token的部分（跳过图像部分）
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)  # 计算交叉熵损失，忽略-100索引

        return logits, loss  # 返回logits和损失

    @torch.no_grad()  # 在生成过程中不计算梯度
    def generate(self, input_ids, image, attention_mask=None, max_new_tokens=5):
        # Process image through vision encoder and projection
        # 通过视觉编码器和投影器处理图像
        image_embd = self.vision_encoder(image)  # 获取图像嵌入
        image_embd = self.MP(image_embd)  # 投影图像嵌入
        
        # Embed initial tokens
        # 嵌入初始token
        token_embd = self.decoder.token_embedding(input_ids)  # 获取输入token的嵌入
        
        # Concatenate image embeddings with token embeddings
        # 将图像嵌入与token嵌入拼接
        combined_embd = torch.cat((image_embd, token_embd), dim=1)  # 拼接图像和token嵌入

        batch_size = image_embd.size(0)  # 获取批次大小
        img_seq_len = image_embd.size(1)  # 获取图像序列长度
        # Adjust attention mask to account for image tokens
        # 调整注意力掩码以考虑图像token
        if attention_mask is not None:
            # Create mask of 1s for image tokens (all image tokens should be attended to)
            # 为图像token创建全1的掩码（所有图像token都应该被关注）
            image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)  # 创建图像注意力掩码
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)  # 合并图像和token掩码
        
        # Generate from combined embeddings using the decoder
        # 使用解码器从合并后的嵌入进行生成
        # We need to use the decoder's forward function and not its generate method
        # because we want to keep track of the image prefix
        # 我们需要使用解码器的forward函数而不是其generate方法，因为我们需要保留图像前缀
        outputs = combined_embd  # 初始化生成序列为合并后的嵌入
        generated_tokens = torch.zeros((batch_size, max_new_tokens), device=input_ids.device, dtype=input_ids.dtype)  # 初始化存储生成token的张量
        
        #Note: Here you could implement improvements like e.g. KV caching
        # 注意：这里可以实现KV缓存等改进
        for i in range(max_new_tokens):  # 循环生成指定数量的新token
            model_out = self.decoder(outputs, attention_mask)  # 通过解码器进行前向传播
            
            # Get predictions for the last token only (normally this is the embedding, not the logits)
            # 只获取最后一个token的预测（通常是嵌入，而不是logits）
            last_token_logits = model_out[:, -1, :]  # 获取最后一个时间步的输出
            
            # Apply head to get logits (if model is in embedding mode)
            # 应用head层获取logits（如果模型处于嵌入模式）
            if not self.decoder.lm_use_tokens:  # 如果decoder输出的是嵌入
                last_token_logits = self.decoder.head(last_token_logits)  # 通过head层转换为logits

            probs = torch.softmax(last_token_logits, dim=-1)  # 计算概率分布
            next_token = torch.multinomial(probs, num_samples=1)  # 从概率分布中采样下一个token
                
            generated_tokens[:, i] = next_token.squeeze(-1)  # 将生成的token存储到结果张量中
            
            # Convert to embedding and append
            # 转换为嵌入并追加
            next_embd = self.decoder.token_embedding(next_token)  # 获取下一个token的嵌入
            outputs = torch.cat((outputs, next_embd), dim=1)  # 将下一个token的嵌入拼接到当前序列
            
            # Update attention mask for the new token
            # 更新注意力掩码以包含新的token
            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device)), dim=1)  # 在注意力掩码后追加一个全1列

        return generated_tokens  # 返回生成的token序列
        
    def load_checkpoint(self, path):
        print(f"Loading weights from full VLM checkpoint: {path}")  # 打印加载检查点的信息
        checkpoint = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # 加载检查点文件，指定设备
        self.load_state_dict(checkpoint)  # 将加载的state_dict加载到模型中

    @classmethod
    def from_pretrained(cls, cfg):
        model = cls(cfg)  # 创建模型实例
        model.vision_encoder = ViT.from_pretrained(cfg)  # 从预训练加载视觉编码器
        model.decoder = LanguageModel.from_pretrained(cfg)  # 从预训练加载语言模型

        return model  # 返回加载了预训练权重的模型实例