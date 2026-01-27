import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from tqdm import tqdm
import os
import dlimp as dl

# 设置 GPU，根据你的情况调整
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 必须确保这一步引用的是你刚刚修改过 dataset.py 的那个包
from octo.data.dataset import make_dataset_from_rlds 

def analyze_task_distribution(dataset_kwargs, top_k=20):
    """
    统计并可视化 RLDS 数据集中的任务分布。
    
    Args:
        dataset_kwargs (dict): 传递给 make_dataset_from_rlds 的参数字典。
        top_k (int): 可视化时显示的出现频率最高的任务数量。
    """
    print(f"正在加载数据集: {dataset_kwargs.get('name')} ...")
    
    # 1. 创建数据集
    # 注意：我们设置 shuffle=False 以便确定性地读取，train=True 读取训练集
    # 这里的关键是只获取 dataset，不需要 dataset_statistics
    dataset, _ = make_dataset_from_rlds(
        **dataset_kwargs, 
        train=True, 
        shuffle=False,
        # 确保包含 language_key，否则无法提取指令
        # 假设你的数据集中语言键名为 'language_instruction' 或 'language'
        # 如果你的 dataset_kwargs 里没有 language_key，请务必在这里添加
    )

    # 2. 遍历数据集并计数
    task_counter = Counter()
    total_trajectories = 0
    
    print("正在遍历数据集统计任务分布 (这可能需要几分钟)...")
    
    # 使用 tqdm 显示进度 (如果知道总长度可以传入 total)
    for traj in tqdm(dataset.iterator()):
        # 提取语言指令
        # 注意：RLDS 中的字符串通常是 bytes 类型，需要解码
        if "language_instruction" in traj["task"]:
            # language_instruction 在 trajectory 级别通常是一个标量字符串
            # 但有时为了对齐可能会被 repeat，我们只取第一个
            lang_instr = traj["task"]["language_instruction"]
            
            # 如果是 Tensor，转换为 numpy 并解码
            if isinstance(lang_instr, tf.Tensor):
                lang_instr = lang_instr.numpy()
            
            # 处理可能的 bytes 类型
            if isinstance(lang_instr, bytes):
                lang_instr = lang_instr.decode('utf-8')
            elif isinstance(lang_instr, np.ndarray): 
                # 如果是数组（因为 repeat），取第一个元素
                if lang_instr.size > 1:
                    lang_instr = lang_instr[0]
                lang_instr = str(lang_instr) if not isinstance(lang_instr, bytes) else lang_instr.decode('utf-8')

            task_counter[lang_instr] += 1
        else:
            task_counter["<No Instruction>"] += 1
            
        total_trajectories += 1

    # 3. 打印统计摘要
    unique_tasks = len(task_counter)
    print("\n" + "="*40)
    print(f"📊 数据集统计摘要")
    print(f"="*40)
    print(f"总轨迹数 (Total Trajectories): {total_trajectories}")
    print(f"独立任务数 (Unique Tasks): {unique_tasks}")
    print(f"="*40)

    # 4. 可视化
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 12))

    # 子图 1: Top-K 任务频率
    plt.subplot(2, 1, 1)
    
    most_common = task_counter.most_common(top_k)
    tasks, counts = zip(*most_common)
    
    sns.barplot(x=list(counts), y=list(tasks), palette="viridis", hue=list(tasks), legend=False)
    plt.title(f"Top {top_k} Most Frequent Tasks (by Trajectory Count)", fontsize=15)
    plt.xlabel("Number of Trajectories")
    plt.ylabel("Language Instruction")

    # 子图 2: 任务频率分布
    plt.subplot(2, 1, 2)
    all_counts = list(task_counter.values())
    sns.histplot(all_counts, bins=50, kde=False, color="skyblue")
    plt.title("Distribution of Trajectory Counts per Task", fontsize=15)
    plt.xlabel("Number of Trajectories per Task")
    plt.ylabel("Number of Unique Tasks")
    plt.yscale('log') 

    plt.tight_layout()
    
    # --- 新增：保存图片 ---
    # save_path 可以是 'task_distribution.png' 或 'task_distribution.pdf'
    save_path = "task_distribution_bridge.png" 
    print(f"正在保存统计图到: {save_path}")
    
    # bbox_inches='tight' 确保长的语言指令不会被边缘裁剪掉
    # dpi=300 保证图片清晰度足够用于论文或报告
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 
    
    plt.show() # 显示图片
    
    # 记得关闭 plot 以释放内存，特别是如果你在循环中画图的话
    plt.close()
    
    return task_counter

def save_detailed_stats(task_counter, filename="task_distribution_bridge.txt"):
    """
    将任务统计结果保存到 txt 文件，方便查阅长尾分布。
    """
    sorted_tasks = task_counter.most_common()
    
    print(f"正在保存任务统计到 {filename} ...")
    
    with open(filename, "w", encoding="utf-8") as f:
        # 写入头部统计
        max_task, max_count = sorted_tasks[0]
        min_task, min_count = sorted_tasks[-1]
        unique_tasks = len(sorted_tasks)
        
        f.write("="*80 + "\n")
        f.write(f"DATASET STATISTICS SUMMARY\n")
        f.write(f"Total Unique Tasks: {unique_tasks}\n")
        f.write(f"Max Trajectories:   {max_count} (Task: {max_task})\n")
        f.write(f"Min Trajectories:   {min_count} (Task: {min_task})\n")
        f.write("="*80 + "\n\n")
        
        # 写入列表
        f.write(f"{'Rank':<6} | {'Count':<8} | {'Language Instruction'}\n")
        f.write("-" * 100 + "\n")
        
        single_traj_count = 0
        
        for rank, (task, count) in enumerate(sorted_tasks, 1):
            f.write(f"{rank:<6} | {count:<8} | {task}\n")
            if count == 1:
                single_traj_count += 1
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"Single Trajectory Tasks: {single_traj_count} / {unique_tasks} ({single_traj_count/unique_tasks:.1%})\n")

    print(f"✅ 保存完成！请查看文件: {filename}")
    print(f"⚠️ 只有 1 条轨迹的任务占比: {single_traj_count/unique_tasks:.1%}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 配置你的数据集参数
    # 请根据实际情况修改 data_dir 和 name
    my_dataset_args = {
        "name": "bridge_dataset",
        "data_dir": "/data/Chenyang/OXE_download",
        "image_obs_keys": {"primary": "image_0"}, 
        "language_key": "language_instruction", # 关键参数
    }

    # 运行分析
    counter = analyze_task_distribution(my_dataset_args)
    sorted_list = save_detailed_stats(counter)