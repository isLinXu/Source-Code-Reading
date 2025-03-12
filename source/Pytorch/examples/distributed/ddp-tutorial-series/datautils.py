import torch  # 导入 PyTorch 库
from torch.utils.data import Dataset  # 从 torch.utils.data 导入 Dataset 基类

class MyTrainDataset(Dataset):  # 定义一个名为 MyTrainDataset 的类，继承自 Dataset
    def __init__(self, size):  # 构造函数，接收数据集大小作为参数
        """Initialize the dataset.
        初始化数据集。
        
        Args:
            size: 数据集的大小
        """
        self.size = size  # 保存数据集的大小
        # 生成指定大小的数据，每个数据点包含一个 20 维的随机张量和一个 1 维的随机张量
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]  

    def __len__(self):  # 返回数据集的大小
        """Return the size of the dataset.
        返回数据集的大小。
        """
        return self.size  # 返回数据集的大小
    
    def __getitem__(self, index):  # 根据索引获取数据点
        """Retrieve a data point by index.
        通过索引获取数据点。
        
        Args:
            index: 数据点的索引
        Returns:
            数据点（包含特征和标签）
        """
        return self.data[index]  # 返回指定索引的数据点