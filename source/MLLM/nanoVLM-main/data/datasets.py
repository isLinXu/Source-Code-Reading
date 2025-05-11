import torch  # 导入PyTorch库
from PIL import Image  # 从PIL库导入Image模块，用于图像处理
from torch.utils.data import Dataset  # 从torch.utils.data导入Dataset基类

import models.config as cfg  # 导入模型配置模块


class VQADataset(Dataset):  # Visual Question Answering Dataset # 视觉问答数据集
    def __init__(self, dataset, tokenizer, image_processor):
        self.dataset = dataset  # 保存原始数据集对象
        self.tokenizer = tokenizer  # 保存tokenizer
        self.image_processor = image_processor  # 保存图像处理器

    def __len__(self):
        return len(self.dataset)  # 返回数据集的样本数量

    def __getitem__(self, idx):
        item = self.dataset[idx]  # 获取指定索引的样本数据

        # Handle image (its a list)
        # 处理图像（它是一个列表）
        image_data = item['images']  # 获取图像数据
        if isinstance(image_data, list) and len(image_data) > 0:  # 如果图像数据是列表且不为空
            image = image_data[0]  # 取列表中的第一个图像
        else:
            image = image_data  # 否则直接使用图像数据

        # Now process the image
        # 现在处理图像
        if isinstance(image, Image.Image):  # 如果图像是PIL Image对象
            if image.mode != 'RGB':  # 如果图像模式不是RGB
                image = image.convert('RGB')  # 转换为RGB模式
            processed_image = self.image_processor(image)  # 使用图像处理器处理图像
        else:
            print(f"Error processing image at index {idx}")  # 打印图像处理错误信息
            # Create empty tensor with right dimensions as fallback
            # 创建具有正确维度的空张量作为备用
            processed_image = torch.zeros(
                3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)  # 创建一个全零张量作为处理失败的备用图像

        # Process text (also a list)
        # 处理文本（也是一个列表）
        text_data = item['texts']  # 获取文本数据
        if isinstance(text_data, list) and len(text_data) > 0:  # 如果文本数据是列表且不为空
            text = text_data[0]  # 取列表中的第一个文本项
        else:
            text = text_data  # 否则直接使用文本数据

        question = text['user']  # 从文本项中提取用户的问题
        # Add EOS token to the answer to train model to predict it, enabling correct stopping during generation
        # 在答案后添加EOS token，以训练模型预测它，从而在生成过程中实现正确的停止
        answer = text['assistant'] + self.tokenizer.eos_token  # 从文本项中提取助手的答案，并在末尾添加EOS token

        formatted_text = f"Question: {question} Answer: "  # 格式化输入文本，包含问题和"Answer: "前缀

        return {
            "image": processed_image,  # 返回处理后的图像张量
            "text_data": formatted_text,  # 返回格式化后的输入文本
            "answer": answer  # 返回包含EOS token的答案文本
        }


class MMStarDataset(Dataset):  # https://huggingface.co/datasets/Lin-Chen/MMStar # MMStar数据集
    def __init__(self, dataset, tokenizer, image_processor):
        self.dataset = dataset  # 保存原始数据集对象
        self.tokenizer = tokenizer  # 保存tokenizer
        self.image_processor = image_processor  # 保存图像处理器
        
    def __len__(self):
        return len(self.dataset)  # 返回数据集的样本数量
    
    def __getitem__(self, idx):
        item = self.dataset[idx]  # 获取指定索引的样本数据
        
        image = item['image']  # 获取图像数据
            
        # Now process the image
        # 现在处理图像
        if isinstance(image, Image.Image):  # 如果图像是PIL Image对象
            if image.mode != 'RGB':  # 如果图像模式不是RGB
                image = image.convert('RGB')  # 转换为RGB模式
            processed_image = self.image_processor(image)  # 使用图像处理器处理图像
        else:
            print(f"Error processing image at index {idx}")  # 打印图像处理错误信息
            # Create empty tensor with right dimensions as fallback
            # 创建具有正确维度的空张量作为备用
            processed_image = torch.zeros(3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)  # 创建一个全零张量作为处理失败的备用图像
        
        question = item['question']  # 从样本中提取问题
        answer = item['answer'] + self.tokenizer.eos_token # Add EOS token to the answer to train model to predict it, enabling correct stopping during generation # 从样本中提取答案，并在末尾添加EOS token，以训练模型预测它，从而在生成过程中实现正确的停止
        
        formatted_text = f"Question: {question} \nAnswer only with the letter! \nAnswer: "  # 格式化输入文本，包含问题和特定的指令以及"Answer: "前缀
        
        return {
            "image": processed_image,  # 返回处理后的图像张量
            "text_data": formatted_text,  # 返回格式化后的输入文本
            "answer": answer  # 返回包含EOS token的答案文本
        }
    