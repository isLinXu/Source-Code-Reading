import re  # 导入正则表达式模块

# Used to check our models performance on multiple choice tasks. This can also be done in a more involved way with e.g. LLM-as-a-judge
# 用于检查模型在多项选择任务上的表现。这也可以通过更复杂的方式完成，例如使用LLM作为评判者。
def check_multiple_choice_with_regex(model_outputs, correct_answers):
    results = []  # 初始化一个列表来存储每个输出的检查结果
    for model_output, correct_answer in zip(model_outputs, correct_answers):  # 遍历模型输出和对应的正确答案
        correct_answer = correct_answer.upper()  # 将正确答案转换为大写，以便进行不区分大小写的匹配

        # Look for the answer letter at the beginning of a line or as the last word
        # 寻找答案字母，可能出现在行首或作为最后一个词
        patterns = [
            rf"\b{correct_answer}\b",  # Word boundary around the answer letter # 答案字母周围有单词边界（确保是独立的词）
            rf"\b{correct_answer}[.,)]",  # Answer followed by punctuation # 答案后跟着标点符号（如句号、逗号、括号）
            rf"\(.*{correct_answer}.*\)",  # Answer within parentheses # 答案出现在括号内
        ]  # 定义一系列正则表达式模式来匹配答案

        match_found = False  # 初始化一个标志，表示是否找到匹配项
        for pattern in patterns:  # 遍历所有定义的模式
            if re.search(pattern, model_output):  # 在模型输出中搜索当前模式
                match_found = True  # 如果找到匹配项，设置标志为True
                break  # Exit inner loop once a match is found # 找到匹配项后，退出内部循环
        results.append(match_found)  # 将当前输出的匹配结果（True或False）添加到结果列表
    return results  # 返回包含所有输出检查结果的列表