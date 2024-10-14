from transformers import AutoConfig


def auto_upgrade(config):
    """
    自动升级配置文件以匹配新的LLaVA代码库。

    当配置文件中包含'LLaVA'且检查点版本为v0时，如果模型类型不是'LLaVA'而是'LLaMA'，
    则提示用户需要升级检查点以匹配新的代码库。用户确认后，自动执行升级过程。

    参数:
    - config: str，配置文件路径或检查点目录。

    返回:
    无返回值，但会根据情况修改并保存配置文件。
    """
    # 从配置或检查点加载AutoConfig
    cfg = AutoConfig.from_pretrained(config)

    # 检查是否需要升级检查点
    if 'llava' in config and 'llava' not in cfg.model_type:
        # 确认用户正在使用旧版本的LLaMA模型
        assert cfg.model_type == 'llama'
        # 提示用户检查点需要升级
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        # 请求用户确认升级
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")

        # 根据用户输入决定是否升级
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            # 确保配置中只有一个架构
            assert len(cfg.architectures) == 1
            # 修改配置的模型类型
            setattr(cfg.__class__, "model_type", "llava")
            # 更新架构类型
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            # 保存更新后的配置
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            # 用户拒绝升级，终止程序
            print("Checkpoint upgrade aborted.")
            exit(1)
