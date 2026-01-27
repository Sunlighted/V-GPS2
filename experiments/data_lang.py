import tensorflow as tf
import os
import numpy as np
from octo.data.dataset import make_dataset_from_rlds 

# 设置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def check_embeddings(dataset_kwargs):
    print(f"正在加载数据集: {dataset_kwargs.get('name')} ...")
    
    dataset, _ = make_dataset_from_rlds(
        **dataset_kwargs, 
        train=True, 
        shuffle=False,
    )

    print("正在检查 Language Embedding 的有效性...")
    
    zero_vector_count = 0
    unique_vectors = set()
    total_count = 0
    
    # 我们只检查前 1000 条，不然太慢
    limit = 1000
    
    for i, traj in enumerate(dataset.iterator()):
        if i >= limit: break
        
        # 提取 embedding
        # 结构通常是: traj['steps']['action']... 但 make_dataset_from_rlds 可能会扁平化
        # 通常 embedding 是一步一个 (512,)，或者整条轨迹共享
        
        emb = None
        
        # 1. 尝试从 task (Metadata) 获取
        if "task" in traj and "language_embedding" in traj["task"]:
            emb = traj["task"]["language_embedding"]
            
        # 2. 尝试从 steps 获取 (每一步都有)
        elif "language_embedding" in traj:
            emb = traj["language_embedding"]
            # 如果是序列，取第一帧
            if len(emb.shape) > 1: 
                emb = emb[0] 
                
        if emb is not None:
            # 转 numpy
            if hasattr(emb, 'numpy'):
                emb = emb.numpy()
            
            # 统计
            total_count += 1
            
            # 检查是否全零
            if np.allclose(emb, 0):
                zero_vector_count += 1
            
            # 检查唯一性 (为了速度，只hash前10位)
            # 把 float 转成 string 来做 hash key
            emb_hash = str(emb[:10]) 
            unique_vectors.add(emb_hash)
            
            if i < 5:
                print(f"Traj {i} Emb First 5 dims: {emb[:5]}")
        else:
            print(f"Traj {i}: 找不到 Embedding 字段")

    print("\n" + "="*50)
    print("📊 Embedding 尸检报告")
    print("="*50)
    print(f"检查样本数: {total_count}")
    print(f"全零向量数 (Zero Vectors): {zero_vector_count}")
    print(f"独立向量数 (Unique Vectors): {len(unique_vectors)}")
    print("="*50)
    
    if len(unique_vectors) < 10 and total_count > 100:
        print("结论: ⚠️ 几乎所有 Embedding 都是一样的！这说明它们大概率来自空字符串。")
    elif zero_vector_count > total_count * 0.5:
        print("结论: ⚠️ 大部分 Embedding 是全零，无效。")
    else:
        print("结论: ✅ Embedding 看起来有多样性，可能比文本字段保存得好？(虽然不太可能)")

if __name__ == "__main__":
    my_dataset_args = {
        "name": "bridge_dataset",
        "data_dir": "/data/Chenyang/OXE_download",
        "image_obs_keys": {"primary": "image_0"}, 
        "language_key": "language_instruction", 
    }

    check_embeddings(my_dataset_args)