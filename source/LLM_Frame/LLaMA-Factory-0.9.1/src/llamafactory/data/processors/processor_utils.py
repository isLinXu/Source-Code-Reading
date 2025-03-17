# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import bisect  # 导入二分查找模块
from typing import List, Sequence, Tuple  # 导入类型提示


def search_for_fit(numbers: Sequence[int], capacity: int) -> int:
    r"""
    Finds the index of largest number that fits into the knapsack with the given capacity.
    查找能装入给定容量背包中的最大数字的索引
    """
    # 使用二分查找找到第一个大于容量的数字的位置
    index = bisect.bisect(numbers, capacity)
    # 如果index为0说明所有数字都大于容量，返回-1
    # 否则返回最后一个小于等于容量的数字的索引
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: List[int], capacity: int) -> List[List[int]]:
    r"""
    An efficient greedy algorithm with binary search for the knapsack problem.
    使用二分查找的高效贪心算法解决背包问题
    """
    numbers.sort()  # sort numbers in ascending order for binary search
                   # 将数字按升序排序以便进行二分查找
    knapsacks = []  # 存储所有背包的列表

    while numbers:  # 当还有数字未处理时
        current_knapsack = []  # 当前背包
        remaining_capacity = capacity  # 剩余容量

        while True:
            # 查找能装入当前剩余容量的最大数字的索引
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:  # 如果没有找到合适的数字
                break  # no more numbers fit in this knapsack
                      # 当前背包无法再装入更多数字

            remaining_capacity -= numbers[index]  # update the remaining capacity
                                                # 更新剩余容量
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack
                                                       # 将数字添加到当前背包并从原列表中移除

        knapsacks.append(current_knapsack)  # 将当前背包添加到背包列表中

    return knapsacks


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:
    r"""
    Computes the real sequence length after truncation by the cutoff_len.
    计算截断后的实际序列长度
    """
    if target_len * 2 < cutoff_len:  # truncate source
                                     # 如果目标长度的两倍小于截断长度，只截断源序列
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
                                      # 如果源长度的两倍小于截断长度，只截断目标序列
        max_target_len = cutoff_len - source_len
    else:  # truncate both
           # 两者都需要截断，按比例分配长度
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    # 计算新的目标长度和源长度
    new_target_len = min(max_target_len, target_len)  # 取较小值作为新的目标长度
    max_source_len = max(cutoff_len - new_target_len, 0)  # 计算源序列的最大可用长度
    new_source_len = min(max_source_len, source_len)  # 取较小值作为新的源长度
    return new_source_len, new_target_len
