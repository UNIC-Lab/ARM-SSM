#!/usr/bin/env python3
"""
分析 EfficientScan 和原始 Mamba 的显存使用差异
"""

import torch
import torch.nn as nn
import sys
import os
import yaml
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from model import RadioMambaNet


def analyze_memory_breakdown(model, input_shape, device, use_efficient_scan):
    """详细分析显存使用情况"""
    model.eval()
    model = model.to(device)
    
    # 创建输入
    x = torch.randn(*input_shape, device=device)
    
    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # 重置统计
        torch.cuda.reset_peak_memory_stats()
        base_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # 模型参数显存
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2  # MB
        print(f"\n{'='*80}")
        print(f"显存分析 - {'EfficientScan' if use_efficient_scan else '原始 Mamba'}")
        print(f"{'='*80}")
        print(f"模型参数显存: {param_memory:.2f} MB")
        print(f"基础显存 (模型加载后): {base_memory:.2f} MB")
        
        # 单次前向传播的峰值显存
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        activation_memory = peak_memory - base_memory
        
        print(f"峰值显存: {peak_memory:.2f} MB")
        print(f"激活值显存 (峰值 - 基础): {activation_memory:.2f} MB")
        print(f"总显存使用: {peak_memory:.2f} MB")
        
        # 计算输入输出大小
        B, C, H, W = input_shape
        input_memory = B * C * H * W * 4 / 1024**2  # float32 = 4 bytes
        print(f"\n输入张量显存: {input_memory:.2f} MB (shape: {input_shape})")
        
        # 估算 EfficientScan 的额外开销
        if use_efficient_scan:
            # EfficientScan 创建 [B, 4, C, H//step_size * W//step_size] 的中间张量
            # 假设 step_size=2
            step_size = 2  # 从配置中获取，这里假设
            new_h = H // step_size
            new_w = W // step_size
            efficient_scan_memory = B * 4 * C * new_h * new_w * 4 / 1024**2
            print(f"\nEfficientScan 中间张量估算:")
            print(f"  xs 张量: [B={B}, 4, C={C}, H//{step_size}*W//{step_size}={new_h}*{new_w}]")
            print(f"  显存: {efficient_scan_memory:.2f} MB")
            print(f"  相对于输入: {efficient_scan_memory/input_memory:.2f}x")
        
        return {
            'param_memory': param_memory,
            'base_memory': base_memory,
            'peak_memory': peak_memory,
            'activation_memory': activation_memory,
            'input_memory': input_memory
        }
    else:
        print("CPU 模式，无法测量显存")
        return None


def load_config(config_path):
    """从 YAML 配置文件加载参数"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config, use_efficient_scan=True, step_size=None):
    """根据配置创建模型"""
    model_config = config.get('Model', config.get('model', {}))
    
    in_channels = model_config.get('in_channels', 3)
    out_channels = model_config.get('out_channels', 1)
    dims = model_config.get('dims', [48, 96, 192, 384])
    depths = model_config.get('depths', [2, 3, 4, 2])
    ssm_d_state = model_config.get('ssm_d_state', 32)
    ssm_d_conv = model_config.get('ssm_d_conv', 4)
    ssm_expand = model_config.get('ssm_expand', 2)
    
    if use_efficient_scan:
        efficient_scan_stage0 = model_config.get('efficient_scan_stage0', True)
        step_size_stage0 = model_config.get('step_size_stage0', 2)
        efficient_scan_stage1 = model_config.get('efficient_scan_stage1', True)
        step_size_stage1 = model_config.get('step_size_stage1', 2)
        efficient_scan_decoder_last = model_config.get('efficient_scan_decoder_last', True)
        step_size_decoder_last = model_config.get('step_size_decoder_last', 2)
        
        if step_size is not None:
            if isinstance(step_size, int):
                step_size_stage0 = step_size
                step_size_stage1 = step_size
                step_size_decoder_last = step_size
            elif isinstance(step_size, dict):
                step_size_stage0 = step_size.get('stage0', step_size_stage0)
                step_size_stage1 = step_size.get('stage1', step_size_stage1)
                step_size_decoder_last = step_size.get('decoder_last', step_size_decoder_last)
    else:
        efficient_scan_stage0 = False
        step_size_stage0 = 1
        efficient_scan_stage1 = False
        step_size_stage1 = 1
        efficient_scan_decoder_last = False
        step_size_decoder_last = 1

    model = RadioMambaNet(
        in_channels=in_channels,
        out_channels=out_channels,
        dims=dims,
        depths=depths,
        ssm_d_state=ssm_d_state,
        ssm_d_conv=ssm_d_conv,
        ssm_expand=ssm_expand,
        efficient_scan_stage0=efficient_scan_stage0,
        step_size_stage0=step_size_stage0,
        efficient_scan_stage1=efficient_scan_stage1,
        step_size_stage1=step_size_stage1,
        efficient_scan_decoder_last=efficient_scan_decoder_last,
        step_size_decoder_last=step_size_decoder_last,
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(description='分析 EfficientScan 显存使用')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--input_size', type=int, default=1024, help='输入图像尺寸')
    parser.add_argument('--step_size', type=int, default=None, help='step_size 值')
    args = parser.parse_args()
    
    config = load_config(args.config)
    model_config = config.get('Model', config.get('model', {}))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = model_config.get('in_channels', 3)
    input_shape = (1, in_channels, args.input_size, args.input_size)
    
    # 解析 step_size
    step_size = args.step_size
    
    print("="*80)
    print("EfficientScan 显存使用分析")
    print("="*80)
    print(f"输入形状: {input_shape}")
    print(f"设备: {device}")
    
    # 分析 EfficientScan 配置
    print("\n" + "="*80)
    model_efficient = create_model_from_config(config, use_efficient_scan=True, step_size=step_size)
    mem_efficient = analyze_memory_breakdown(model_efficient, input_shape, device, use_efficient_scan=True)
    
    # 清理
    del model_efficient
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # 分析原始 Mamba 配置
    print("\n" + "="*80)
    model_original = create_model_from_config(config, use_efficient_scan=False)
    mem_original = analyze_memory_breakdown(model_original, input_shape, device, use_efficient_scan=False)
    
    # 对比
    if mem_efficient and mem_original:
        print("\n" + "="*80)
        print("对比结果")
        print("="*80)
        print(f"\n参数量显存差异:")
        print(f"  EfficientScan: {mem_efficient['param_memory']:.2f} MB")
        print(f"  原始 Mamba: {mem_original['param_memory']:.2f} MB")
        print(f"  差异: {mem_efficient['param_memory'] - mem_original['param_memory']:.2f} MB")
        print(f"  倍数: {mem_efficient['param_memory'] / mem_original['param_memory']:.4f}x")
        
        print(f"\n激活值显存差异:")
        print(f"  EfficientScan: {mem_efficient['activation_memory']:.2f} MB")
        print(f"  原始 Mamba: {mem_original['activation_memory']:.2f} MB")
        print(f"  差异: {mem_efficient['activation_memory'] - mem_original['activation_memory']:.2f} MB")
        print(f"  倍数: {mem_efficient['activation_memory'] / mem_original['activation_memory']:.4f}x")
        
        print(f"\n总显存差异:")
        print(f"  EfficientScan: {mem_efficient['peak_memory']:.2f} MB")
        print(f"  原始 Mamba: {mem_original['peak_memory']:.2f} MB")
        print(f"  差异: {mem_efficient['peak_memory'] - mem_original['peak_memory']:.2f} MB")
        print(f"  倍数: {mem_efficient['peak_memory'] / mem_original['peak_memory']:.4f}x")
        
        # 分析原因
        print(f"\n原因分析:")
        param_diff = mem_efficient['param_memory'] - mem_original['param_memory']
        activation_diff = mem_efficient['activation_memory'] - mem_original['activation_memory']
        total_diff = mem_efficient['peak_memory'] - mem_original['peak_memory']
        
        print(f"  参数显存增加: {param_diff:.2f} MB ({param_diff/total_diff*100:.1f}%)")
        print(f"  激活值显存增加: {activation_diff:.2f} MB ({activation_diff/total_diff*100:.1f}%)")
        print(f"  总增加: {total_diff:.2f} MB")
        
        if param_diff > 0:
            print(f"\n  ⚠️  EfficientScan 配置有更多参数，导致参数显存增加")
        if activation_diff > 0:
            print(f"  ⚠️  EfficientScan 在 forward 过程中创建了额外的中间张量")
            print(f"      (如 EfficientScan 的 4 个子图张量 xs: [B, 4, C, H//step_size * W//step_size])")
    
    print("\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == "__main__":
    main()

