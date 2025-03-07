# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import concurrent.futures  # 导入并发执行模块
import statistics  # 导入统计模块
import time  # 导入时间模块
from typing import List, Optional, Tuple  # 从 typing 模块导入类型注解

import requests  # 导入请求模块


class GCPRegions:
    """
    A class for managing and analyzing Google Cloud Platform (GCP) regions.  # 管理和分析 Google Cloud Platform (GCP) 区域的类

    This class provides functionality to initialize, categorize, and analyze GCP regions based on their  # 此类提供功能以初始化、分类和分析 GCP 区域
    geographical location, tier classification, and network latency.  # 基于地理位置、等级分类和网络延迟。

    Attributes:
        regions (Dict[str, Tuple[int, str, str]]): A dictionary of GCP regions with their tier, city, and country.  # 包含 GCP 区域及其等级、城市和国家的字典。

    Methods:
        tier1: Returns a list of tier 1 GCP regions.  # 返回等级 1 的 GCP 区域列表。
        tier2: Returns a list of tier 2 GCP regions.  # 返回等级 2 的 GCP 区域列表。
        lowest_latency: Determines the GCP region(s) with the lowest network latency.  # 确定网络延迟最低的 GCP 区域。

    Examples:
        >>> from ultralytics.hub.google import GCPRegions  # 从 ultralytics.hub.google 导入 GCPRegions
        >>> regions = GCPRegions()  # 创建 GCPRegions 实例
        >>> lowest_latency_region = regions.lowest_latency(verbose=True, attempts=3)  # 获取最低延迟的区域
        >>> print(f"Lowest latency region: {lowest_latency_region[0][0]}")  # 打印最低延迟区域的名称
    """

    def __init__(self):
        """Initializes the GCPRegions class with predefined Google Cloud Platform regions and their details."""  # 用预定义的 Google Cloud Platform 区域及其详细信息初始化 GCPRegions 类
        self.regions = {  # 定义区域及其等级、城市和国家
            "asia-east1": (1, "Taiwan", "China"),  # 台湾
            "asia-east2": (2, "Hong Kong", "China"),  # 香港
            "asia-northeast1": (1, "Tokyo", "Japan"),  # 东京
            "asia-northeast2": (1, "Osaka", "Japan"),  # 大阪
            "asia-northeast3": (2, "Seoul", "South Korea"),  # 首尔
            "asia-south1": (2, "Mumbai", "India"),  # 孟买
            "asia-south2": (2, "Delhi", "India"),  # 新德里
            "asia-southeast1": (2, "Jurong West", "Singapore"),  # 裕廊西
            "asia-southeast2": (2, "Jakarta", "Indonesia"),  # 雅加达
            "australia-southeast1": (2, "Sydney", "Australia"),  # 悉尼
            "australia-southeast2": (2, "Melbourne", "Australia"),  # 墨尔本
            "europe-central2": (2, "Warsaw", "Poland"),  # 华沙
            "europe-north1": (1, "Hamina", "Finland"),  # 哈米纳
            "europe-southwest1": (1, "Madrid", "Spain"),  # 马德里
            "europe-west1": (1, "St. Ghislain", "Belgium"),  # 圣吉斯兰
            "europe-west10": (2, "Berlin", "Germany"),  # 柏林
            "europe-west12": (2, "Turin", "Italy"),  # 都灵
            "europe-west2": (2, "London", "United Kingdom"),  # 伦敦
            "europe-west3": (2, "Frankfurt", "Germany"),  # 法兰克福
            "europe-west4": (1, "Eemshaven", "Netherlands"),  # 埃门哈芬
            "europe-west6": (2, "Zurich", "Switzerland"),  # 苏黎世
            "europe-west8": (1, "Milan", "Italy"),  # 米兰
            "europe-west9": (1, "Paris", "France"),  # 巴黎
            "me-central1": (2, "Doha", "Qatar"),  # 多哈
            "me-west1": (1, "Tel Aviv", "Israel"),  # 特拉维夫
            "northamerica-northeast1": (2, "Montreal", "Canada"),  # 蒙特利尔
            "northamerica-northeast2": (2, "Toronto", "Canada"),  # 多伦多
            "southamerica-east1": (2, "São Paulo", "Brazil"),  # 圣保罗
            "southamerica-west1": (2, "Santiago", "Chile"),  # 圣地亚哥
            "us-central1": (1, "Iowa", "United States"),  # 爱荷华州
            "us-east1": (1, "South Carolina", "United States"),  # 南卡罗来纳州
            "us-east4": (1, "Northern Virginia", "United States"),  # 北弗吉尼亚州
            "us-east5": (1, "Columbus", "United States"),  # 哥伦布
            "us-south1": (1, "Dallas", "United States"),  # 达拉斯
            "us-west1": (1, "Oregon", "United States"),  # 俄勒冈州
            "us-west2": (2, "Los Angeles", "United States"),  # 洛杉矶
            "us-west3": (2, "Salt Lake City", "United States"),  # 盐湖城
            "us-west4": (2, "Las Vegas", "United States"),  # 拉斯维加斯
        }

    def tier1(self) -> List[str]:
        """Returns a list of GCP regions classified as tier 1 based on predefined criteria."""  # 返回根据预定义标准分类为等级 1 的 GCP 区域列表
        return [region for region, info in self.regions.items() if info[0] == 1]  # 从 regions 中筛选等级 1 的区域

    def tier2(self) -> List[str]:
        """Returns a list of GCP regions classified as tier 2 based on predefined criteria."""  # 返回根据预定义标准分类为等级 2 的 GCP 区域列表
        return [region for region, info in self.regions.items() if info[0] == 2]  # 从 regions 中筛选等级 2 的区域

    @staticmethod
    def _ping_region(region: str, attempts: int = 1) -> Tuple[str, float, float, float, float]:
        """Pings a specified GCP region and returns latency statistics: mean, min, max, and standard deviation."""  # Ping 指定的 GCP 区域并返回延迟统计信息：均值、最小值、最大值和标准差
        url = f"https://{region}-docker.pkg.dev"  # 构建 URL
        latencies = []  # 初始化延迟列表
        for _ in range(attempts):  # 根据尝试次数进行循环
            try:
                start_time = time.time()  # 记录开始时间
                _ = requests.head(url, timeout=5)  # 发送 HEAD 请求
                latency = (time.time() - start_time) * 1000  # convert latency to milliseconds  # 将延迟转换为毫秒
                if latency != float("inf"):  # 如果延迟不是无穷大
                    latencies.append(latency)  # 将延迟添加到列表
            except requests.RequestException:  # 捕获请求异常
                pass  # 忽略异常

        if not latencies:  # 如果没有记录到延迟
            return region, float("inf"), float("inf"), float("inf"), float("inf")  # 返回无穷大延迟

        std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0  # 计算标准差
        return region, statistics.mean(latencies), std_dev, min(latencies), max(latencies)  # 返回区域及其延迟统计信息

    def lowest_latency(
        self,
        top: int = 1,
        verbose: bool = False,
        tier: Optional[int] = None,
        attempts: int = 1,
    ) -> List[Tuple[str, float, float, float, float]]:
        """
        Determines the GCP regions with the lowest latency based on ping tests.  # 根据 ping 测试确定延迟最低的 GCP 区域

        Args:
            top (int): Number of top regions to return.  # 返回的前 N 个区域的数量
            verbose (bool): If True, prints detailed latency information for all tested regions.  # 如果为 True，则打印所有测试区域的详细延迟信息
            tier (int | None): Filter regions by tier (1 or 2). If None, all regions are tested.  # 按等级过滤区域（1 或 2）。如果为 None，则测试所有区域
            attempts (int): Number of ping attempts per region.  # 每个区域的 ping 尝试次数

        Returns:
            (List[Tuple[str, float, float, float, float]]): List of tuples containing region information and  # 返回包含区域信息和延迟统计信息的元组列表
            latency statistics. Each tuple contains (region, mean_latency, std_dev, min_latency, max_latency).  # 每个元组包含（区域、平均延迟、标准差、最小延迟、最大延迟）

        Examples:
            >>> regions = GCPRegions()  # 创建 GCPRegions 实例
            >>> results = regions.lowest_latency(top=3, verbose=True, tier=1, attempts=2)  # 获取最低延迟的区域
            >>> print(results[0][0])  # Print the name of the lowest latency region  # 打印最低延迟区域的名称
        """
        if verbose:  # 如果需要详细信息
            print(f"Testing GCP regions for latency (with {attempts} {'retry' if attempts == 1 else 'attempts'})...")  # 打印测试信息

        regions_to_test = [k for k, v in self.regions.items() if v[0] == tier] if tier else list(self.regions.keys())  # 根据等级筛选区域
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:  # 创建线程池
            results = list(executor.map(lambda r: self._ping_region(r, attempts), regions_to_test))  # 并发 ping 区域

        sorted_results = sorted(results, key=lambda x: x[1])  # 根据延迟排序结果

        if verbose:  # 如果需要详细信息
            print(f"{'Region':<25} {'Location':<35} {'Tier':<5} Latency (ms)")  # 打印表头
            for region, mean, std, min_, max_ in sorted_results:  # 遍历排序后的结果
                tier, city, country = self.regions[region]  # 获取区域的等级、城市和国家
                location = f"{city}, {country}"  # 构建位置字符串
                if mean == float("inf"):  # 如果平均延迟为无穷大
                    print(f"{region:<25} {location:<35} {tier:<5} Timeout")  # 打印超时信息
                else:
                    print(f"{region:<25} {location:<35} {tier:<5} {mean:.0f} ± {std:.0f} ({min_:.0f} - {max_:.0f})")  # 打印延迟信息
            print(f"\nLowest latency region{'s' if top > 1 else ''}:")  # 打印最低延迟区域信息
            for region, mean, std, min_, max_ in sorted_results[:top]:  # 打印前 N 个最低延迟区域
                tier, city, country = self.regions[region]  # 获取区域的等级、城市和国家
                location = f"{city}, {country}"  # 构建位置字符串
                print(f"{region} ({location}, {mean:.0f} ± {std:.0f} ms ({min_:.0f} - {max_:.0f}))")  # 打印区域信息

        return sorted_results[:top]  # 返回前 N 个最低延迟区域


# Usage example
if __name__ == "__main__":  # 如果是主程序
    regions = GCPRegions()  # 创建 GCPRegions 实例
    top_3_latency_tier1 = regions.lowest_latency(top=3, verbose=True, tier=1, attempts=3)  # 获取等级 1 的最低延迟前 3 个区域