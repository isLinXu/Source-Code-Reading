import torch  # 导入PyTorch库

class VAQCollator(object):  # Visual Question Answering Collator # 视觉问答数据整理器
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer  # 保存tokenizer
        self.max_length = max_length  # 保存最大序列长度
    
    def __call__(self, batch):
        images = [item["image"] for item in batch]  # 从批次中提取图像列表
        texts = [item["text_data"] for item in batch]  # 从批次中提取文本（问题）列表
        answers = [item["answer"] for item in batch]  # 从批次中提取答案列表

        # Stack images
        # 堆叠图像
        images = torch.stack(images)  # 将图像列表堆叠成一个张量

        # Create inputs by concatenating the question and answer
        # 通过拼接问题和答案创建输入序列
        input_sequences = []  # 初始化输入序列列表
        for i in range(len(texts)):  # 遍历每个样本
            input_sequences.append(f"{texts[i]} {answers[i]}")  # 将问题和答案拼接起来作为输入序列

        encoded_full_sequences = self.tokenizer.batch_encode_plus(
            input_sequences,  # 输入序列列表
            padding="max_length",  # 填充到最大长度
            padding_side="left",  # 在左侧进行填充
            return_tensors="pt",  # 返回PyTorch张量
            truncation=True,  # 启用截断
            max_length=self.max_length,  # 截断和填充的最大长度
        )

        # Create labels where only answer tokens are predicted
        # 创建标签，只对答案token进行预测
        input_ids = encoded_full_sequences["input_ids"]  # 获取编码后的输入token ID
        attention_mask = encoded_full_sequences["attention_mask"]  # 获取注意力掩码
        labels = input_ids.clone()  # 复制input_ids作为初始标签
        labels[:, :-1] = input_ids[:, 1:].clone()  # 将标签向左移动一位，实现因果语言建模（预测下一个token）
        labels[:, -1] = -100 #self.tokenizer.pad_token_id # 将最后一个token的标签设置为-100，表示忽略预测（或者设置为pad_token_id）

        # The tokenizer has different behavior for padding and truncation:
        # 1. If the full text (answer + question) is shorter than the max length, its gets padded on the left
        # 2. If the full text is longer than the max length, it gets truncated on the right
        # Therefore, I need to handle multipe cases, this is the different scenarios:
        # If the full text is longer than the max lenght, we need to set the labels to -100 for the whole sample (we want to ignore the whole sample)
        # If the full text is shorter than the max lenght, we need to set the labels to -100 only for the question part, and create causal language modeling labels for the answer part, taking into account the padding

        # tokenizer对于填充和截断有不同的行为：
        # 1. 如果完整文本（答案+问题）短于最大长度，它会在左侧填充
        # 2. 如果完整文本长于最大长度，它会在右侧截断
        # 因此，我需要处理多种情况，这是不同的场景：
        # 如果完整文本长于最大长度，我们需要将整个样本的标签设置为-100（我们想完全忽略这个样本）
        # 如果完整文本短于最大长度，我们需要只将问题部分的标签设置为-100，并为答案部分创建因果语言建模标签，同时考虑填充

        # Determine if sequences were truncated
        # 判断序列是否被截断
        original_lengths = [len(self.tokenizer.encode(seq)) for seq in input_sequences]  # 计算原始序列的长度
        
        for i in range(len(batch)):  # 遍历批次中的每个样本
            # Get the length of the question for this sample
            # 获取这个样本的问题长度
            question_length = len(self.tokenizer.encode(texts[i], add_special_tokens=False))  # 计算问题的token长度，不添加特殊token
            
            # Case 1: If sequence was truncated (original is longer than max_length)
            # 情况1：如果序列被截断（原始长度长于最大长度）
            if original_lengths[i] > self.max_length:
                # Set all labels to -100 to ignore this sample entirely
                # 将所有标签设置为-100，以完全忽略这个样本
                labels[i, :] = -100  # 将当前样本的所有标签设置为-100
                #print(f"Sample {i} was truncated. Setting all labels to -100.") # 打印截断信息（注释掉）
                continue  # 跳过当前样本，处理下一个
            
            # Case 2: Sequence fits within max_length
            # 情况2：序列长度在最大长度范围内
            # Use attention mask to find first non-padding token
            # 使用注意力掩码找到第一个非填充token
            # The first 1 in the attention mask marks the first non-padding token
            # 注意力掩码中的第一个1标记了第一个非填充token
            first_token_pos = attention_mask[i].nonzero(as_tuple=True)[0][0].item()  # 找到注意力掩码中第一个1的位置，即第一个非填充token的位置
            
            # Set labels for padding and question part to -100 (don't predict these), substracting 1 to account for the left shift
            # 将填充和问题部分的标签设置为-100（不预测这些），减去1是为了考虑左移
            question_end = first_token_pos + question_length - 1  # 计算问题结束的位置（考虑填充和左移）
            labels[i, :question_end] = -100  # 将从开始到问题结束位置的标签设置为-100

        return {
            "image": images,  # 返回图像张量
            "input_ids": input_ids,  # 返回输入token ID张量
            "attention_mask": attention_mask,  # 返回注意力掩码张量
            "labels": labels  # 返回标签张量
        }

class MMStarCollator(object):  # https://huggingface.co/datasets/Lin-Chen/MMStar # MMStar数据集数据整理器
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer  # 保存tokenizer
    
    def __call__(self, batch):
        images = [item["image"] for item in batch]  # 从批次中提取图像列表
        questions = [item["text_data"] for item in batch]  # 从批次中提取问题列表
        answers = [item["answer"] for item in batch]  # 从批次中提取答案列表

        # Stack images
        # 堆叠图像
        images = torch.stack(images)  # 将图像列表堆叠成一个张量
        
        encoded_question_sequences = self.tokenizer.batch_encode_plus(
            questions,  # 问题列表
            padding=True,  # 启用填充
            padding_side="left",  # 在左侧进行填充
            return_tensors="pt"  # 返回PyTorch张量
        )

        encoded_answer_sequences = self.tokenizer.batch_encode_plus(
            answers,  # 答案列表
            padding=True,  # 启用填充
            padding_side="left",  # 在左侧进行填充
            return_tensors="pt"  # 返回PyTorch张量
        )
        
        return {
            "images": images,  # 返回图像张量
            "input_ids": encoded_question_sequences['input_ids'],  # 返回编码后的问题token ID张量
            "attention_mask": encoded_question_sequences['attention_mask'],  # 返回问题的注意力掩码张量
            "labels": encoded_answer_sequences['input_ids'],  # 返回编码后的答案token ID张量（作为标签用于评估）
        }