# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import re  # 导入re模块，用于正则表达式操作
import shutil  # 导入shutil模块，用于文件和目录的操作
import subprocess  # 导入subprocess模块，用于执行子进程
from itertools import repeat  # 从itertools导入repeat，用于重复元素
from multiprocessing.pool import ThreadPool  # 从multiprocessing导入ThreadPool，用于多线程池
from pathlib import Path  # 从pathlib导入Path，用于路径操作
from urllib import parse, request  # 从urllib导入parse和request模块，用于URL解析和请求

import requests  # 导入requests模块，用于HTTP请求
import torch  # 导入torch模块，用于深度学习

from ultralytics.utils import LOGGER, TQDM, checks, clean_url, emojis, is_online, url2file  # 从utils模块导入各种工具函数

# Define Ultralytics GitHub assets maintained at https://github.com/ultralytics/assets
# 定义Ultralytics GitHub资产，维护在https://github.com/ultralytics/assets
GITHUB_ASSETS_REPO = "ultralytics/assets"  # GitHub资产库名称
GITHUB_ASSETS_NAMES = (
    [f"yolov8{k}{suffix}.pt" for k in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose", "-obb", "-oiv7")]
    + [f"yolo11{k}{suffix}.pt" for k in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose", "-obb")]
    + [f"yolov5{k}{resolution}u.pt" for k in "nsmlx" for resolution in ("", "6")]
    + [f"yolov3{k}u.pt" for k in ("", "-spp", "-tiny")]
    + [f"yolov8{k}-world.pt" for k in "smlx"]
    + [f"yolov8{k}-worldv2.pt" for k in "smlx"]
    + [f"yolov9{k}.pt" for k in "tsmce"]
    + [f"yolov10{k}.pt" for k in "nsmblx"]
    + [f"yolo_nas_{k}.pt" for k in "sml"]
    + [f"sam_{k}.pt" for k in "bl"]
    + [f"FastSAM-{k}.pt" for k in "sx"]
    + [f"rtdetr-{k}.pt" for k in "lx"]
    + ["mobile_sam.pt"]  # 移动SAM模型
    + ["calibration_image_sample_data_20x128x128x3_float32.npy.zip"]  # 校准图像样本数据
)
GITHUB_ASSETS_STEMS = [Path(k).stem for k in GITHUB_ASSETS_NAMES]  # 获取资产名称的基本名称

def is_url(url, check=False):
    """
    Validates if the given string is a URL and optionally checks if the URL exists online.
    验证给定字符串是否为URL，并可选择性地检查该URL是否在线存在。

    Args:
        url (str): The string to be validated as a URL.
        url (str): 要验证为URL的字符串。
        check (bool, optional): If True, performs an additional check to see if the URL exists online.
            Defaults to False.
        check (bool, optional): 如果为True，则执行额外检查以查看URL是否在线存在。默认为False。

    Returns:
        (bool): Returns True for a valid URL. If 'check' is True, also returns True if the URL exists online.
            Returns False otherwise.
        (bool): 对于有效的URL返回True。如果'check'为True，则如果URL在线存在也返回True。否则返回False。

    Example:
        ```python
        valid = is_url("https://www.example.com")
        ```
    """
    try:
        url = str(url)  # 将URL转换为字符串
        result = parse.urlparse(url)  # 解析URL
        assert all([result.scheme, result.netloc])  # check if is url 检查是否为URL
        if check:  # 如果需要检查
            with request.urlopen(url) as response:  # 打开URL并获取响应
                return response.getcode() == 200  # check if exists online 检查是否在线存在
        return True  # URL有效
    except Exception:  # 捕获异常
        return False  # URL无效


def delete_dsstore(path, files_to_delete=(".DS_Store", "__MACOSX")):
    """
    Deletes all ".DS_store" files under a specified directory.
    删除指定目录下的所有“.DS_store”文件。

    Args:
        path (str, optional): The directory path where the ".DS_store" files should be deleted.
        path (str, optional): 要删除“.DS_store”文件的目录路径。
        files_to_delete (tuple): The files to be deleted.
        files_to_delete (tuple): 要删除的文件。

    Example:
        ```python
        from ultralytics.utils.downloads import delete_dsstore

        delete_dsstore("path/to/dir")
        ```

    Note:
        ".DS_store" files are created by the Apple operating system and contain metadata about folders and files. They
        are hidden system files and can cause issues when transferring files between different operating systems.
        ".DS_store"文件由苹果操作系统创建，包含有关文件夹和文件的元数据。它们是隐藏的系统文件，在不同操作系统之间传输文件时可能会导致问题。
    """
    for file in files_to_delete:  # 遍历要删除的文件
        matches = list(Path(path).rglob(file))  # 查找匹配的文件
        LOGGER.info(f"Deleting {file} files: {matches}")  # 记录删除信息
        for f in matches:  # 遍历匹配的文件
            f.unlink()  # 删除文件


def zip_directory(directory, compress=True, exclude=(".DS_Store", "__MACOSX"), progress=True):
    """
    Zips the contents of a directory, excluding files containing strings in the exclude list. The resulting zip file is
    named after the directory and placed alongside it.
    压缩目录的内容，排除包含排除列表中字符串的文件。生成的zip文件以目录命名并放置在其旁边。

    Args:
        directory (str | Path): The path to the directory to be zipped.
        directory (str | Path): 要压缩的目录的路径。
        compress (bool): Whether to compress the files while zipping. Default is True.
        compress (bool): 是否在压缩时压缩文件。默认为True。
        exclude (tuple, optional): A tuple of filename strings to be excluded. Defaults to ('.DS_Store', '__MACOSX').
        exclude (tuple, optional): 要排除的文件名字符串元组。默认为('.DS_Store', '__MACOSX')。
        progress (bool, optional): Whether to display a progress bar. Defaults to True.
        progress (bool, optional): 是否显示进度条。默认为True。

    Returns:
        (Path): The path to the resulting zip file.
        (Path): 结果zip文件的路径。

    Example:
        ```python
        from ultralytics.utils.downloads import zip_directory

        file = zip_directory("path/to/dir")
        ```
    """
    from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile  # 从zipfile导入压缩相关的类

    delete_dsstore(directory)  # 删除目录中的.DS_Store文件
    directory = Path(directory)  # 将目录转换为Path对象
    if not directory.is_dir():  # 检查目录是否存在
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")  # 抛出文件未找到异常

    # Unzip with progress bar
    files_to_zip = [f for f in directory.rglob("*") if f.is_file() and all(x not in f.name for x in exclude)]  # 查找要压缩的文件
    zip_file = directory.with_suffix(".zip")  # 创建zip文件的路径
    compression = ZIP_DEFLATED if compress else ZIP_STORED  # 设置压缩方式
    with ZipFile(zip_file, "w", compression) as f:  # 创建zip文件
        for file in TQDM(files_to_zip, desc=f"Zipping {directory} to {zip_file}...", unit="file", disable=not progress):  # 显示进度条
            f.write(file, file.relative_to(directory))  # 将文件写入zip

    return zip_file  # return path to zip file 返回zip文件路径


def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX"), exist_ok=False, progress=True):
    """
    Unzips a *.zip file to the specified path, excluding files containing strings in the exclude list.
    解压缩*.zip文件到指定路径，排除包含排除列表中字符串的文件。

    If the zipfile does not contain a single top-level directory, the function will create a new
    directory with the same name as the zipfile (without the extension) to extract its contents.
    如果zip文件不包含单个顶级目录，函数将创建一个与zip文件同名的新目录（不带扩展名）以提取其内容。
    If a path is not provided, the function will use the parent directory of the zipfile as the default path.
    如果未提供路径，函数将使用zip文件的父目录作为默认路径。

    Args:
        file (str | Path): The path to the zipfile to be extracted.
        file (str | Path): 要提取的zip文件的路径。
        path (str, optional): The path to extract the zipfile to. Defaults to None.
        path (str, optional): 提取zip文件的路径。默认为None。
        exclude (tuple, optional): A tuple of filename strings to be excluded. Defaults to ('.DS_Store', '__MACOSX').
        exclude (tuple, optional): 要排除的文件名字符串元组。默认为('.DS_Store', '__MACOSX')。
        exist_ok (bool, optional): Whether to overwrite existing contents if they exist. Defaults to False.
        exist_ok (bool, optional): 如果存在，是否覆盖现有内容。默认为False。
        progress (bool, optional): Whether to display a progress bar. Defaults to True.
        progress (bool, optional): 是否显示进度条。默认为True。

    Raises:
        BadZipFile: If the provided file does not exist or is not a valid zipfile.
        BadZipFile: 如果提供的文件不存在或不是有效的zip文件。

    Returns:
        (Path): The path to the directory where the zipfile was extracted.
        (Path): 提取zip文件的目录路径。

    Example:
        ```python
        from ultralytics.utils.downloads import unzip_file

        dir = unzip_file("path/to/file.zip")
        ```
    """
    from zipfile import BadZipFile, ZipFile, is_zipfile  # 从zipfile导入相关类

    if not (Path(file).exists() and is_zipfile(file)):  # 检查文件是否存在并且是zip文件
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")  # 抛出异常
    if path is None:  # 如果未提供路径
        path = Path(file).parent  # default path 默认路径

    # Unzip the file contents
    with ZipFile(file) as zipObj:  # 打开zip文件
        files = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]  # 获取要提取的文件
        top_level_dirs = {Path(f).parts[0] for f in files}  # 获取顶级目录

        # Decide to unzip directly or unzip into a directory
        unzip_as_dir = len(top_level_dirs) == 1  # (len(files) > 1 and not files[0].endswith("/")) 判断是否只有一个顶级目录
        if unzip_as_dir:  # 如果只有一个顶级目录
            extract_path = path  # 提取路径
            path = Path(path) / list(top_level_dirs)[0]  # 将内容提取到指定目录
        else:  # 如果有多个文件在顶层
            path = extract_path = Path(path) / Path(file).stem  # 提取到新目录

        # Check if destination directory already exists and contains files
        if path.exists() and any(path.iterdir()) and not exist_ok:  # 检查目标目录是否存在且不为空
            # If it exists and is not empty, return the path without unzipping
            LOGGER.warning(f"WARNING ⚠️ Skipping {file} unzip as destination directory {path} is not empty.")  # 记录警告
            return path  # 返回路径

        for f in TQDM(files, desc=f"Unzipping {file} to {Path(path).resolve()}...", unit="file", disable=not progress):  # 显示进度条
            # Ensure the file is within the extract_path to avoid path traversal security vulnerability
            if ".." in Path(f).parts:  # 检查路径是否安全
                LOGGER.warning(f"Potentially insecure file path: {f}, skipping extraction.")  # 记录警告
                continue  # 跳过提取
            zipObj.extract(f, extract_path)  # 提取文件

    return path  # return unzip dir 返回解压目录


def check_disk_space(url="https://ultralytics.com/assets/coco8.zip", path=Path.cwd(), sf=1.5, hard=True):
    """
    Check if there is sufficient disk space to download and store a file.
    检查是否有足够的磁盘空间来下载和存储文件。

    Args:
        url (str, optional): The URL to the file. Defaults to 'https://ultralytics.com/assets/coco8.zip'.
        url (str, optional): 文件的URL。默认为'https://ultralytics.com/assets/coco8.zip'。
        path (str | Path, optional): The path or drive to check the available free space on.
        path (str | Path, optional): 检查可用空间的路径或驱动器。
        sf (float, optional): Safety factor, the multiplier for the required free space. Defaults to 1.5.
        sf (float, optional): 安全因子，所需可用空间的乘数。默认为1.5。
        hard (bool, optional): Whether to throw an error or not on insufficient disk space. Defaults to True.
        hard (bool, optional): 磁盘空间不足时是否抛出错误。默认为True。

    Returns:
        (bool): True if there is sufficient disk space, False otherwise.
        (bool): 如果有足够的磁盘空间则返回True，否则返回False。
    """
    try:
        r = requests.head(url)  # response 获取响应
        assert r.status_code < 400, f"URL error for {url}: {r.status_code} {r.reason}"  # 检查响应状态
    except Exception:  # 捕获异常
        return True  # requests issue, default to True 请求问题，默认返回True

    # Check file size
    gib = 1 << 30  # bytes per GiB 每GiB的字节数
    data = int(r.headers.get("Content-Length", 0)) / gib  # file size (GB) 文件大小（GB）
    total, used, free = (x / gib for x in shutil.disk_usage(path))  # bytes 计算总、已用和可用空间

    if data * sf < free:  # 如果可用空间大于所需空间
        return True  # sufficient space 有足够的空间

    # Insufficient space
    text = (
        f"WARNING ⚠️ Insufficient free disk space {free:.1f} GB < {data * sf:.3f} GB required, "
        f"Please free {data * sf - free:.1f} GB additional disk space and try again."
    )  # 创建警告信息
    if hard:  # 如果需要抛出错误
        raise MemoryError(text)  # 抛出内存错误
    LOGGER.warning(text)  # 记录警告
    return False  # 返回False


def get_google_drive_file_info(link):
    """
    Retrieves the direct download link and filename for a shareable Google Drive file link.
    获取可共享Google Drive文件链接的直接下载链接和文件名。

    Args:
        link (str): The shareable link of the Google Drive file.
        link (str): Google Drive文件的共享链接。

    Returns:
        (str): Direct download URL for the Google Drive file.
        (str): Google Drive文件的直接下载URL。
        (str): Original filename of the Google Drive file. If filename extraction fails, returns None.
        (str): Google Drive文件的原始文件名。如果文件名提取失败，则返回None。

    Example:
        ```python
        from ultralytics.utils.downloads import get_google_drive_file_info

        link = "https://drive.google.com/file/d/1cqT-cJgANNrhIHCrEufUYhQ4RqiWG_lJ/view?usp=drive_link"
        url, filename = get_google_drive_file_info(link)
        ```
    """
    file_id = link.split("/d/")[1].split("/view")[0]  # 从链接中提取文件ID
    drive_url = f"https://drive.google.com/uc?export=download&id={file_id}"  # 创建直接下载链接
    filename = None  # 初始化文件名

    # Start session
    with requests.Session() as session:  # 创建会话
        response = session.get(drive_url, stream=True)  # 获取响应流
        if "quota exceeded" in str(response.content.lower()):  # 检查配额是否超限
            raise ConnectionError(
                emojis(
                    f"❌  Google Drive file download quota exceeded. "
                    f"Please try again later or download this file manually at {link}."
                )
            )  # 抛出连接错误
        for k, v in response.cookies.items():  # 遍历cookie
            if k.startswith("download_warning"):  # 检查下载警告
                drive_url += f"&confirm={v}"  # v是token
        if cd := response.headers.get("content-disposition"):  # 获取内容处置头
            filename = re.findall('filename="(.+)"', cd)[0]  # 提取文件名
    return drive_url, filename  # 返回下载链接和文件名


def safe_download(
    url,
    file=None,
    dir=None,
    unzip=True,
    delete=False,
    curl=False,
    retry=3,
    min_bytes=1e0,
    exist_ok=False,
    progress=True,
):
    """
    Downloads files from a URL, with options for retrying, unzipping, and deleting the downloaded file.
    从URL下载文件，支持重试、解压和删除下载文件的选项。

    Args:
        url (str): The URL of the file to be downloaded.
        url (str): 要下载的文件的URL。
        file (str, optional): The filename of the downloaded file.
            If not provided, the file will be saved with the same name as the URL.
        file (str, optional): 下载文件的文件名。如果未提供，将使用URL的相同名称保存文件。
        dir (str, optional): The directory to save the downloaded file.
            If not provided, the file will be saved in the current working directory.
        dir (str, optional): 保存下载文件的目录。如果未提供，文件将保存在当前工作目录中。
        unzip (bool, optional): Whether to unzip the downloaded file. Default: True.
        unzip (bool, optional): 是否解压下载的文件。默认为True。
        delete (bool, optional): Whether to delete the downloaded file after unzipping. Default: False.
        delete (bool, optional): 解压后是否删除下载的文件。默认为False。
        curl (bool, optional): Whether to use curl command line tool for downloading. Default: False.
        curl (bool, optional): 是否使用curl命令行工具下载。默认为False。
        retry (int, optional): The number of times to retry the download in case of failure. Default: 3.
        retry (int, optional): 失败时重试下载的次数。默认为3。
        min_bytes (float, optional): The minimum number of bytes that the downloaded file should have, to be considered
            a successful download. Default: 1E0.
        min_bytes (float, optional): 下载文件应具有的最小字节数，以被视为成功下载。默认为1E0。
        exist_ok (bool, optional): Whether to overwrite existing contents during unzipping. Defaults to False.
        exist_ok (bool, optional): 解压时是否覆盖现有内容。默认为False。
        progress (bool, optional): Whether to display a progress bar during the download. Default: True.
        progress (bool, optional): 下载过程中是否显示进度条。默认为True。

    Example:
        ```python
        from ultralytics.utils.downloads import safe_download

        link = "https://ultralytics.com/assets/bus.jpg"
        path = safe_download(link)
        ```
    """
    gdrive = url.startswith("https://drive.google.com/")  # check if the URL is a Google Drive link 检查URL是否为Google Drive链接
    if gdrive:  # 如果是Google Drive链接
        url, file = get_google_drive_file_info(url)  # 获取直接下载链接和文件名

    f = Path(dir or ".") / (file or url2file(url))  # URL转换为文件名
    if "://" not in str(url) and Path(url).is_file():  # URL存在（Windows Python <3.10需要检查'://'）
        f = Path(url)  # 文件名
    elif not f.is_file():  # URL和文件都不存在
        uri = (url if gdrive else clean_url(url)).replace(  # cleaned and aliased url 清理和别名URL
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/",
            "https://ultralytics.com/assets/",  # assets alias 资产别名
        )
        desc = f"Downloading {uri} to '{f}'"  # 下载描述
        LOGGER.info(f"{desc}...")  # 记录下载信息
        f.parent.mkdir(parents=True, exist_ok=True)  # 如果缺失则创建目录
        check_disk_space(url, path=f.parent)  # 检查磁盘空间
        for i in range(retry + 1):  # 重试次数
            try:
                if curl or i > 0:  # curl下载重试，继续
                    s = "sS" * (not progress)  # silent 静默模式
                    r = subprocess.run(["curl", "-#", f"-{s}L", url, "-o", f, "--retry", "3", "-C", "-"]).returncode  # 使用curl下载
                    assert r == 0, f"Curl return value {r}"  # 检查curl返回值
                else:  # urllib下载
                    method = "torch"  # 下载方法
                    if method == "torch":  # 使用torch下载
                        torch.hub.download_url_to_file(url, f, progress=progress)  # 下载文件
                    else:  # 使用urllib下载
                        with request.urlopen(url) as response, TQDM(
                            total=int(response.getheader("Content-Length", 0)),  # 获取内容长度
                            desc=desc,  # 下载描述
                            disable=not progress,  # 是否禁用进度条
                            unit="B",  # 单位为字节
                            unit_scale=True,  # 启用单位缩放
                            unit_divisor=1024,  # 单位除数
                        ) as pbar:
                            with open(f, "wb") as f_opened:  # 打开文件以写入
                                for data in response:  # 遍历响应数据
                                    f_opened.write(data)  # 写入数据
                                    pbar.update(len(data))  # 更新进度条

                if f.exists():  # 如果文件存在
                    if f.stat().st_size > min_bytes:  # 检查文件大小
                        break  # success 成功
                    f.unlink()  # remove partial downloads 删除部分下载
            except Exception as e:  # 捕获异常
                if i == 0 and not is_online():  # 如果是第一次重试且不在线
                    raise ConnectionError(emojis(f"❌  Download failure for {uri}. Environment is not online.")) from e  # 抛出连接错误
                elif i >= retry:  # 如果达到重试次数
                    raise ConnectionError(emojis(f"❌  Download failure for {uri}. Retry limit reached.")) from e  # 抛出连接错误
                LOGGER.warning(f"⚠️ Download failure, retrying {i + 1}/{retry} {uri}...")  # 记录下载失败并重试

    if unzip and f.exists() and f.suffix in {"", ".zip", ".tar", ".gz"}:  # 如果需要解压且文件存在且后缀有效
        from zipfile import is_zipfile  # 导入is_zipfile函数

        unzip_dir = (dir or f.parent).resolve()  # 解压到提供的目录，如果未提供则解压到当前目录
        if is_zipfile(f):  # 如果是zip文件
            unzip_dir = unzip_file(file=f, path=unzip_dir, exist_ok=exist_ok, progress=progress)  # 解压文件
        elif f.suffix in {".tar", ".gz"}:  # 如果是tar或gz文件
            LOGGER.info(f"Unzipping {f} to {unzip_dir}...")  # 记录解压信息
            subprocess.run(["tar", "xf" if f.suffix == ".tar" else "xfz", f, "--directory", unzip_dir], check=True)  # 解压文件
        if delete:  # 如果需要删除
            f.unlink()  # remove zip 删除zip文件
        return unzip_dir  # 返回解压目录

def get_github_assets(repo="ultralytics/assets", version="latest", retry=False):
    """
    Retrieve the specified version's tag and assets from a GitHub repository. If the version is not specified, the
    function fetches the latest release assets.
    从GitHub仓库获取指定版本的标签和资产。如果未指定版本，则函数获取最新发布的资产。

    Args:
        repo (str, optional): The GitHub repository in the format 'owner/repo'. Defaults to 'ultralytics/assets'.
        repo (str, optional): GitHub仓库，格式为'owner/repo'。默认为'ultralytics/assets'。
        version (str, optional): The release version to fetch assets from. Defaults to 'latest'.
        version (str, optional): 要获取资产的发布版本。默认为'latest'。
        retry (bool, optional): Flag to retry the request in case of a failure. Defaults to False.
        retry (bool, optional): 如果请求失败，是否重试。默认为False。

    Returns:
        (tuple): A tuple containing the release tag and a list of asset names.
        (tuple): 返回一个元组，包含发布标签和资产名称列表。

    Example:
        ```python
        tag, assets = get_github_assets(repo="ultralytics/assets", version="latest")
        ```
    """
    if version != "latest":  # 如果版本不是最新的
        version = f"tags/{version}"  # i.e. tags/v6.2 例如：tags/v6.2
    url = f"https://api.github.com/repos/{repo}/releases/{version}"  # 构建API请求URL
    r = requests.get(url)  # github api 获取GitHub API响应
    if r.status_code != 200 and r.reason != "rate limit exceeded" and retry:  # 如果请求失败且不是403速率限制且允许重试
        r = requests.get(url)  # 尝试再次请求
    if r.status_code != 200:  # 如果请求仍然失败
        LOGGER.warning(f"⚠️ GitHub assets check failure for {url}: {r.status_code} {r.reason}")  # 记录警告信息
        return "", []  # 返回空标签和空资产列表
    data = r.json()  # 解析响应为JSON格式
    return data["tag_name"], [x["name"] for x in data["assets"]]  # 返回标签和资产名称列表，例如：['yolo11n.pt', 'yolov8s.pt', ...]


def attempt_download_asset(file, repo="ultralytics/assets", release="v8.3.0", **kwargs):
    """
    Attempt to download a file from GitHub release assets if it is not found locally. The function checks for the file
    locally first, then tries to download it from the specified GitHub repository release.
    如果在本地未找到文件，则尝试从GitHub发布资产下载文件。该函数首先检查本地文件，然后尝试从指定的GitHub仓库发布下载。

    Args:
        file (str | Path): The filename or file path to be downloaded.
        file (str | Path): 要下载的文件名或文件路径。
        repo (str, optional): The GitHub repository in the format 'owner/repo'. Defaults to 'ultralytics/assets'.
        repo (str, optional): GitHub仓库，格式为'owner/repo'。默认为'ultralytics/assets'。
        release (str, optional): The specific release version to be downloaded. Defaults to 'v8.3.0'.
        release (str, optional): 要下载的特定发布版本。默认为'v8.3.0'。
        **kwargs (any): Additional keyword arguments for the download process.
        **kwargs (any): 下载过程的其他关键字参数。

    Returns:
        (str): The path to the downloaded file.
        (str): 下载文件的路径。

    Example:
        ```python
        file_path = attempt_download_asset("yolo11n.pt", repo="ultralytics/assets", release="latest")
        ```
    """
    from ultralytics.utils import SETTINGS  # scoped for circular import 作用域用于循环导入

    # YOLOv3/5u updates
    file = str(file)  # 将文件转换为字符串
    file = checks.check_yolov5u_filename(file)  # 检查YOLOv5u文件名的有效性
    file = Path(file.strip().replace("'", ""))  # 去掉文件名两端的空格和单引号
    if file.exists():  # 如果文件在当前路径下存在
        return str(file)  # 返回文件路径
    elif (SETTINGS["weights_dir"] / file).exists():  # 如果文件在权重目录下存在
        return str(SETTINGS["weights_dir"] / file)  # 返回权重目录下的文件路径
    else:
        # URL specified
        name = Path(parse.unquote(str(file))).name  # decode '%2F' to '/' etc. 解码URL中的特殊字符
        download_url = f"https://github.com/{repo}/releases/download"  # 构建下载URL
        if str(file).startswith(("http:/", "https:/")):  # 如果文件名以http或https开头
            url = str(file).replace(":/", "://")  # Pathlib将://转换为:/
            file = url2file(name)  # 解析认证 https://url.com/file.txt?auth...
            if Path(file).is_file():  # 如果文件已经存在
                LOGGER.info(f"Found {clean_url(url)} locally at {file}")  # 记录找到的文件信息
            else:
                safe_download(url=url, file=file, min_bytes=1e5, **kwargs)  # 下载文件

        elif repo == GITHUB_ASSETS_REPO and name in GITHUB_ASSETS_NAMES:  # 如果是GitHub资产库并且文件名在资产名称列表中
            safe_download(url=f"{download_url}/{release}/{name}", file=file, min_bytes=1e5, **kwargs)  # 下载文件

        else:
            tag, assets = get_github_assets(repo, release)  # 获取指定版本的标签和资产
            if not assets:  # 如果没有资产
                tag, assets = get_github_assets(repo)  # 获取最新发布的资产
            if name in assets:  # 如果文件名在资产列表中
                safe_download(url=f"{download_url}/{tag}/{name}", file=file, min_bytes=1e5, **kwargs)  # 下载文件

        return str(file)  # 返回文件路径


def download(url, dir=Path.cwd(), unzip=True, delete=False, curl=False, threads=1, retry=3, exist_ok=False):
    """
    Downloads files from specified URLs to a given directory. Supports concurrent downloads if multiple threads are
    specified.
    从指定的URL下载文件到给定目录。如果指定了多个线程，则支持并发下载。

    Args:
        url (str | list): The URL or list of URLs of the files to be downloaded.
        url (str | list): 要下载的文件的URL或URL列表。
        dir (Path, optional): The directory where the files will be saved. Defaults to the current working directory.
        dir (Path, optional): 文件保存的目录。默认为当前工作目录。
        unzip (bool, optional): Flag to unzip the files after downloading. Defaults to True.
        unzip (bool, optional): 下载后是否解压文件。默认为True。
        delete (bool, optional): Flag to delete the zip files after extraction. Defaults to False.
        delete (bool, optional): 解压后是否删除zip文件。默认为False。
        curl (bool, optional): Flag to use curl for downloading. Defaults to False.
        curl (bool, optional): 是否使用curl进行下载。默认为False。
        threads (int, optional): Number of threads to use for concurrent downloads. Defaults to 1.
        threads (int, optional): 用于并发下载的线程数。默认为1。
        retry (int, optional): Number of retries in case of download failure. Defaults to 3.
        retry (int, optional): 下载失败时的重试次数。默认为3。
        exist_ok (bool, optional): Whether to overwrite existing contents during unzipping. Defaults to False.
        exist_ok (bool, optional): 解压时是否覆盖现有内容。默认为False。

    Example:
        ```python
        download("https://ultralytics.com/assets/example.zip", dir="path/to/dir", unzip=True)
        ```
    """
    dir = Path(dir)  # 将目录转换为Path对象
    dir.mkdir(parents=True, exist_ok=True)  # 创建目录（如果不存在）
    if threads > 1:  # 如果线程数大于1
        with ThreadPool(threads) as pool:  # 创建线程池
            pool.map(
                lambda x: safe_download(
                    url=x[0],  # 下载的URL
                    dir=x[1],  # 保存的目录
                    unzip=unzip,
                    delete=delete,
                    curl=curl,
                    retry=retry,
                    exist_ok=exist_ok,
                    progress=threads <= 1,  # 如果线程数小于等于1，则显示进度条
                ),
                zip(url, repeat(dir)),  # 将URL和目录组合
            )
            pool.close()  # 关闭线程池
            pool.join()  # 等待所有线程完成
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:  # 如果URL是字符串或Path对象
            safe_download(url=u, dir=dir, unzip=unzip, delete=delete, curl=curl, retry=retry, exist_ok=exist_ok)  # 下载文件