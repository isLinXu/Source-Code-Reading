import os  # 导入os模块，用于与操作系统交互
import zipfile  # 导入zipfile模块，用于处理ZIP文件

# PyTorch 1.1 moves _download_url_to_file  # PyTorch 1.1将_download_url_to_file移动
#   from torch.utils.model_zoo to torch.hub  # 从torch.utils.model_zoo移动到torch.hub
# PyTorch 1.0 exists another _download_url_to_file  # PyTorch 1.0存在另一个_download_url_to_file
#   2 argument  # 2个参数
# TODO: If you remove support PyTorch 1.0 or older,  # TODO: 如果你移除对PyTorch 1.0或更早版本的支持，
#       You should remove torch.utils.model_zoo  # 应该移除torch.utils.model_zoo
#       Ref. PyTorch #18758  # 参考: PyTorch #18758
#         https://github.com/pytorch/pytorch/pull/18758/commits  # 链接到相关提交
try:
    from torch.utils.model_zoo import _download_url_to_file  # 尝试从torch.utils.model_zoo导入_download_url_to_file
except ImportError:  # 如果导入失败
    try:
        from torch.hub import download_url_to_file as _download_url_to_file  # 尝试从torch.hub导入download_url_to_file
    except ImportError:  # 如果再次导入失败
        from torch.hub import _download_url_to_file  # 从torch.hub导入_download_url_to_file


def unzip(source_filename, dest_dir):  # 定义解压缩函数
    with zipfile.ZipFile(source_filename) as zf:  # 打开ZIP文件
        zf.extractall(path=dest_dir)  # 解压缩到指定目录


if __name__ == '__main__':  # 如果是主模块
    _download_url_to_file('https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=1', 'saved_models.zip', None, True)  # 下载ZIP文件
    unzip('saved_models.zip', '.')  # 解压缩下载的ZIP文件到当前目录