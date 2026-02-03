import json
import numpy as np
from collections import defaultdict

def calculate_hash_collision_rate(hash_data):
    """
    计算哈希碰撞率
    :param hash_data: 输入字典，格式如 {"哈希值": [[时间, 标签], ...], ...}
    :return: 碰撞统计结果字典
    """
    # 初始化统计变量
    stats = {
        "总哈希键数": 0,
        "总锚点记录数": 0,
        "冗余重复记录数": 0,  # 同一哈希+同一时间+同一标签的重复项
        "真碰撞哈希数": 0,    # 同一哈希+不同时间/标签的哈希数
        "去重后总记录数": 0
    }

    # 遍历每个哈希键
    for hash_key, records in hash_data.items():
        stats["总哈希键数"] += 1
        original_count = len(records)
        stats["总锚点记录数"] += original_count

        # 1. 去重（统计冗余重复）
        # 转元组才能哈希去重（列表不可哈希）
        unique_records = list(set(tuple(item) for item in records))
        unique_count = len(unique_records)
        stats["去重后总记录数"] += unique_count
        stats["冗余重复记录数"] += (original_count - unique_count)

        # 2. 统计真哈希碰撞（同一哈希对应不同记录）
        if unique_count > 1:
            stats["真碰撞哈希数"] += 1

    # 计算碰撞率
    stats["冗余重复率"] = stats["冗余重复记录数"] / stats["总锚点记录数"] * 100 if stats["总锚点记录数"] > 0 else 0
    stats["真哈希碰撞率"] = stats["真碰撞哈希数"] / stats["总哈希键数"] * 100 if stats["总哈希键数"] > 0 else 0

    return stats

def print_collision_report(stats):
    """打印可视化的碰撞分析报告"""
    print("="*50)
    print("            哈希碰撞率分析报告")
    print("="*50)
    print(f"1. 基础统计：")
    print(f"   - 总哈希键数量：{stats['总哈希键数']:,}")
    print(f"   - 总锚点记录数量：{stats['总锚点记录数']:,}")
    print(f"   - 去重后记录数量：{stats['去重后总记录数']:,}")
    print(f"\n2. 冗余重复分析（同一哈希+同一时间+标签）：")
    print(f"   - 冗余重复记录数：{stats['冗余重复记录数']:,}")
    print(f"   - 冗余重复率：{stats['冗余重复率']:.2f}%")
    print(f"\n3. 真哈希碰撞分析（同一哈希+不同时间/标签）：")
    print(f"   - 真碰撞哈希键数：{stats['真碰撞哈希数']:,}")
    print(f"   - 真哈希碰撞率：{stats['真哈希碰撞率']:.2f}%")
    print("="*50)

# ===================== 示例调用（适配你的数据） =====================
if __name__ == "__main__":


    with open("generate_V3_data.db", "r", encoding="utf-8") as f:
        your_hash_data = json.load(f)

    # 计算碰撞率
    collision_stats = calculate_hash_collision_rate(your_hash_data)
    # 打印报告
    print_collision_report(collision_stats)