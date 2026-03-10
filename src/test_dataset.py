#!/usr/bin/env python3
"""
测试 LocationDataset 数据集加载
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader
from dataset import LocationDataset
import numpy as np

# 可选依赖
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib 未安装，将跳过可视化部分")

def test_dataset():
    print("=" * 80)
    print("测试 LocationDataset 数据集加载")
    print("=" * 80)
    
    # 创建数据集实例
    print("\n1. 创建数据集实例...")
    try:
        # 先检查文件解析
        import os
        pathloss_dir = "/home/zqm1/dataset/Location/location_512/pathloss_process"
        pathloss_files = sorted([f for f in os.listdir(pathloss_dir) if f.endswith('.png')])
        print(f"   - 找到 {len(pathloss_files)} 个 pathloss 文件")
        if len(pathloss_files) > 0:
            print(f"   - 示例文件: {pathloss_files[:3]}")
        
        dataset = LocationDataset(
            phase="train",
            dir_pathloss="/home/zqm1/dataset/Location/location_512/pathloss_process",
            dir_buildings="/home/zqm1/dataset/Location/cond/buildings_complete",
            dir_tx="/home/zqm1/dataset/Location/cond/xy_pngs",
        )
        print(f"   ✓ 数据集创建成功")
        print(f"   - 数据集大小: {len(dataset)}")
        print(f"   - 图像尺寸: {dataset.height}x{dataset.width}")
        print(f"   - Building 索引范围: [{dataset.ind1}, {dataset.ind2}]")
        print(f"   - 唯一 buildings 数量: {len(dataset.maps_inds)}")
    except Exception as e:
        print(f"   ✗ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试单个样本加载
    print("\n2. 测试单个样本加载...")
    try:
        sample_idx = 0
        inputs, target, filename = dataset[sample_idx]
        
        print(f"   ✓ 样本 {sample_idx} 加载成功")
        print(f"   - 文件名: {filename}")
        print(f"   - 输入形状: {inputs.shape}")
        print(f"   - 目标形状: {target.shape}")
        print(f"   - 输入数据类型: {inputs.dtype}")
        print(f"   - 目标数据类型: {target.dtype}")
        print(f"   - 输入值范围: [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"   - 目标值范围: [{target.min():.3f}, {target.max():.3f}]")
        
        # 检查第三个通道的采样率
        third_channel = inputs[2, :, :]  # 第三个通道
        non_zero_pixels = (third_channel != 0).sum().item()
        total_pixels = third_channel.numel()
        sampling_rate = non_zero_pixels / total_pixels * 100
        print(f"   - 第三个通道采样率: {sampling_rate:.2f}% ({non_zero_pixels}/{total_pixels})")
        
    except Exception as e:
        print(f"   ✗ 样本加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试多个样本
    print("\n3. 测试多个样本加载...")
    try:
        num_test_samples = min(5, len(dataset))
        for i in range(num_test_samples):
            inputs, target, filename = dataset[i]
            print(f"   ✓ 样本 {i}: {filename}, 输入形状: {inputs.shape}, 目标形状: {target.shape}")
        print(f"   ✓ 成功加载 {num_test_samples} 个样本")
    except Exception as e:
        print(f"   ✗ 多样本加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试 DataLoader
    print("\n4. 测试 DataLoader...")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0  # 使用 0 避免多进程问题
        )
        
        batch = next(iter(dataloader))
        inputs_batch, targets_batch, filenames_batch = batch
        
        print(f"   ✓ DataLoader 创建成功")
        print(f"   - Batch 输入形状: {inputs_batch.shape}")
        print(f"   - Batch 目标形状: {targets_batch.shape}")
        print(f"   - Batch 大小: {len(filenames_batch)}")
        print(f"   - 文件名: {filenames_batch}")
        
    except Exception as e:
        print(f"   ✗ DataLoader 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试不同 phase
    print("\n5. 测试不同 phase...")
    try:
        train_dataset = LocationDataset(phase="train")
        val_dataset = LocationDataset(phase="val")
        test_dataset = LocationDataset(phase="test")
        
        print(f"   ✓ Train 数据集大小: {len(train_dataset)}")
        print(f"   ✓ Val 数据集大小: {len(val_dataset)}")
        print(f"   ✓ Test 数据集大小: {len(test_dataset)}")
        print(f"   ✓ 总数据集大小: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
        
    except Exception as e:
        print(f"   ✗ Phase 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 可视化一个样本（可选）
    print("\n6. 可视化样本（保存到文件）...")
    if not HAS_MATPLOTLIB:
        print("   ⚠ 跳过可视化（matplotlib 未安装）")
    else:
        try:
            inputs, target, filename = dataset[0]
            
            # 计算采样率
            third_channel = inputs[2, :, :]
            non_zero_pixels = (third_channel != 0).sum().item()
            total_pixels = third_channel.numel()
            sampling_rate = non_zero_pixels / total_pixels * 100
            
            # 转换为 numpy 用于可视化
            inputs_np = inputs.numpy()
            target_np = target.numpy()
            
            # 创建可视化图像
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 输入通道
            axes[0, 0].imshow(inputs_np[0], cmap='gray')
            axes[0, 0].set_title('Channel 1: Buildings')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(inputs_np[1], cmap='gray')
            axes[0, 1].set_title('Channel 2: Tx Location')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(inputs_np[2], cmap='gray')
            axes[0, 2].set_title('Channel 3: Sampled Pathloss (1%)')
            axes[0, 2].axis('off')
            
            # 目标
            axes[1, 0].imshow(target_np[0], cmap='gray')
            axes[1, 0].set_title('Target: Full Pathloss')
            axes[1, 0].axis('off')
            
            # 合并输入（RGB 可视化）
            rgb_input = np.stack([inputs_np[0], inputs_np[1], inputs_np[2]], axis=0)
            rgb_input = np.transpose(rgb_input, (1, 2, 0))
            rgb_input = np.clip(rgb_input, 0, 1)
            axes[1, 1].imshow(rgb_input)
            axes[1, 1].set_title('Combined Input (RGB)')
            axes[1, 1].axis('off')
            
            # 对比
            axes[1, 2].axis('off')
            axes[1, 2].text(0.5, 0.5, f'Filename: {filename}\n'
                                       f'Input shape: {inputs.shape}\n'
                                       f'Target shape: {target.shape}\n'
                                       f'Sampling rate: {sampling_rate:.2f}%',
                            ha='center', va='center', fontsize=12)
            
            plt.tight_layout()
            save_path = '/home/zqm1/workspace2/RadioMamba-main/src/test_dataset_sample.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ 可视化图像已保存到: {save_path}")
            plt.close()
            
        except Exception as e:
            print(f"   ⚠ 可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    test_dataset()

