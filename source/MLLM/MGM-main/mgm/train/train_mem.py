from mgm.train.train import train  
# 导入 mgm.train.train 模块中的 train 函数

if __name__ == "__main__":  
# 判断当前脚本是否作为主程序执行
    train(attn_implementation="flash_attention_2")  
    # 调用 train 函数，并传入参数 attn_implementation="flash_attention_2"，指定使用 flash_attention_2 实现