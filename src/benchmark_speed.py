#!/usr/bin/env python3
"""
完整的 RadioMambaNet 模型性能对比脚本
对比使用 EfficientScan 配置和不使用 EfficientScan 配置的完整模型
从配置文件加载参数
"""

import torch
import torch.nn as nn
import time
import sys
import os
import yaml
import argparse
"""
单次测试示例:
python benchmark_speed.py --config ../configs/config_nocars_1024_effcient.yaml --input_size 1024
python benchmark_speed.py --config ../configs/config_nocars_512_effcient.yaml --input_size 512
python benchmark_speed.py --config ../configs/config_nocars_512_effcient.yaml --input_size 512 --step_size 4
python benchmark_speed.py --config ../configs/config_nocars_512_effcient.yaml --input_size 512 --step_size_stage0 4 --step_size_stage1 2 --step_size_decoder_last 2
python benchmark_speed.py --config ../configs/config_nocars_512_effcient.yaml --input_size 1024 --step_size_stage0 4 --step_size_stage1 2 --step_size_decoder_last 2

批量测试不同 stage0 step_size (固定 stage1=2, decoder_last=2):
python benchmark_stage0_sweep.py --config ../configs/config_nocars_512_effcient.yaml --input_size 1024 --step_size_stage0_list 2 4  --include_baseline
python benchmark_stage0_sweep.py --config ../configs/config_nocars_512_effcient.yaml --input_size 512 --step_size_stage0_list 1 2 4 --include_baseline --output_csv results_stage0_sweep.csv
"""

sys.path.insert(0, os.path.dirname(__file__))

# 导入完整模型
from model import RadioMambaNet


def benchmark_model(model, input_shape, device, num_iterations=50):
    """测试完整模型的性能 - 使用 CUDA Events 精确测量 GPU 执行时间"""
    model.eval()
    model = model.to(device)
    
    # 创建输入（数据已在 GPU 上，无传输开销）
    x = torch.randn(*input_shape, device=device)
    
    # 预热（增加预热次数，确保 GPU 状态稳定）
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # 同步，确保预热完成
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # 使用 CUDA Events 精确测量 GPU 执行时间
    if device.type == 'cuda':
        # 创建 CUDA Events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # 开始测量
        start_event.record()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
        end_event.record()
        
        # 等待 GPU 完成所有操作
        torch.cuda.synchronize()
        
        # 获取 GPU 执行时间（毫秒）
        total_gpu_time_ms = start_event.elapsed_time(end_event)
        avg_time = total_gpu_time_ms / num_iterations
    else:
        # CPU 模式：使用 time.time()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations * 1000  # ms
    
    # 测试显存
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(x)
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    else:
        peak_memory = 0
    
    return avg_time, peak_memory


def load_config(config_path):
    """从 YAML 配置文件加载参数"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config, use_efficient_scan=True, step_size=None):
    """根据配置创建模型
    
    Args:
        config: 配置文件字典
        use_efficient_scan: 是否使用 EfficientScan
        step_size: 可选的 step_size 值，如果提供则覆盖配置文件中的值
                   可以是单个整数（应用于所有阶段）或字典 {'stage0': int, 'stage1': int, 'decoder_last': int}
    """
    model_config = config.get('Model', config.get('model', {}))
    
    # 提取模型参数
    in_channels = model_config.get('in_channels', 3)
    out_channels = model_config.get('out_channels', 1)
    dims = model_config.get('dims', [48, 96, 192, 384])
    depths = model_config.get('depths', [2, 3, 4, 2])
    ssm_d_state = model_config.get('ssm_d_state', 32)
    ssm_d_conv = model_config.get('ssm_d_conv', 4)
    ssm_expand = model_config.get('ssm_expand', 2)
    
    # EfficientScan 配置
    if use_efficient_scan:
        efficient_scan_stage0 = model_config.get('efficient_scan_stage0', True)
        step_size_stage0 = model_config.get('step_size_stage0', 2)
        efficient_scan_stage1 = model_config.get('efficient_scan_stage1', True)
        step_size_stage1 = model_config.get('step_size_stage1', 2)
        efficient_scan_decoder_last = model_config.get('efficient_scan_decoder_last', True)
        step_size_decoder_last = model_config.get('step_size_decoder_last', 2)
        
        # 如果提供了 step_size 参数，覆盖配置文件中的值
        if step_size is not None:
            if isinstance(step_size, int):
                # 单个整数：应用于所有阶段
                step_size_stage0 = step_size
                step_size_stage1 = step_size
                step_size_decoder_last = step_size
            elif isinstance(step_size, dict):
                # 字典：分别设置各个阶段
                step_size_stage0 = step_size.get('stage0', step_size_stage0)
                step_size_stage1 = step_size.get('stage1', step_size_stage1)
                step_size_decoder_last = step_size.get('decoder_last', step_size_decoder_last)
    else:
        # 全部禁用 EfficientScan
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


def print_efficient_scan_config(model_config, use_efficient_scan, step_size=None):
    """打印 EfficientScan 配置信息
    
    Args:
        model_config: 模型配置字典
        use_efficient_scan: 是否使用 EfficientScan
        step_size: 可选的 step_size 值，如果提供则显示实际使用的值
    """
    if use_efficient_scan:
        print(f"  EfficientScan 配置:")
        
        # 确定实际使用的 step_size 值
        if step_size is not None:
            if isinstance(step_size, int):
                # 单个值应用于所有阶段
                actual_step_size_stage0 = step_size
                actual_step_size_stage1 = step_size
                actual_step_size_decoder_last = step_size
            elif isinstance(step_size, dict):
                # 字典分别设置
                actual_step_size_stage0 = step_size.get('stage0', model_config.get('step_size_stage0', 2))
                actual_step_size_stage1 = step_size.get('stage1', model_config.get('step_size_stage1', 2))
                actual_step_size_decoder_last = step_size.get('decoder_last', model_config.get('step_size_decoder_last', 2))
            else:
                # 使用配置文件中的值
                actual_step_size_stage0 = model_config.get('step_size_stage0', 2)
                actual_step_size_stage1 = model_config.get('step_size_stage1', 2)
                actual_step_size_decoder_last = model_config.get('step_size_decoder_last', 2)
        else:
            # 使用配置文件中的值
            actual_step_size_stage0 = model_config.get('step_size_stage0', 2)
            actual_step_size_stage1 = model_config.get('step_size_stage1', 2)
            actual_step_size_decoder_last = model_config.get('step_size_decoder_last', 2)
        
        print(f"    Stage 0: {model_config.get('efficient_scan_stage0', True)} (step_size={actual_step_size_stage0})")
        print(f"    Stage 1: {model_config.get('efficient_scan_stage1', True)} (step_size={actual_step_size_stage1})")
        print(f"    Decoder Last: {model_config.get('efficient_scan_decoder_last', True)} (step_size={actual_step_size_decoder_last})")
    else:
        print(f"  EfficientScan: 全部禁用 (使用原始 Mamba)")


def main():
    parser = argparse.ArgumentParser(description='RadioMambaNet 完整模型性能对比')
    parser.add_argument('--config', type=str, 
                       default='configs/config_nocars.yaml',
                       help='配置文件路径')
    parser.add_argument('--input_size', type=int, default=512,
                       help='输入图像尺寸 (默认: 512)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='批次大小 (默认: 1)')
    parser.add_argument('--iterations', type=int, default=50,
                       help='测试迭代次数 (默认: 50)')
    parser.add_argument('--step_size', type=int, default=None,
                       help='EfficientScan 的 step_size 值，应用于所有阶段 (默认: 使用配置文件中的值)')
    parser.add_argument('--step_size_stage0', type=int, default=None,
                       help='Stage 0 的 step_size (默认: 使用配置文件中的值)')
    parser.add_argument('--step_size_stage1', type=int, default=None,
                       help='Stage 1 的 step_size (默认: 使用配置文件中的值)')
    parser.add_argument('--step_size_decoder_last', type=int, default=None,
                       help='Decoder Last 的 step_size (默认: 使用配置文件中的值)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("RadioMambaNet 完整模型性能对比测试")
    print("使用 CUDA Events 精确测量 GPU 执行时间")
    print("=" * 80)
    
    # 加载配置
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    print(f"\n加载配置文件: {config_path}")
    config = load_config(config_path)
    model_config = config.get('Model', config.get('model', {}))
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 测试配置
    in_channels = model_config.get('in_channels', 3)
    batch_size = args.batch_size
    H, W = args.input_size, args.input_size
    input_shape = (batch_size, in_channels, H, W)
    
    print(f"\n测试配置:")
    print(f"  输入形状: {input_shape}")
    print(f"  dims: {model_config.get('dims', [48, 96, 192, 384])}")
    print(f"  depths: {model_config.get('depths', [2, 3, 4, 2])}")
    print(f"  ssm_d_state: {model_config.get('ssm_d_state', 32)}")
    print(f"  ssm_d_conv: {model_config.get('ssm_d_conv', 4)}")
    print(f"  ssm_expand: {model_config.get('ssm_expand', 2)}")
    
    # 解析 step_size 参数
    step_size = None
    if args.step_size is not None:
        # 如果提供了 --step_size，应用于所有阶段
        step_size = args.step_size
        print(f"  step_size (所有阶段): {step_size}")
    elif any([args.step_size_stage0, args.step_size_stage1, args.step_size_decoder_last]):
        # 如果提供了分别的参数，构建字典
        step_size = {}
        if args.step_size_stage0 is not None:
            step_size['stage0'] = args.step_size_stage0
        if args.step_size_stage1 is not None:
            step_size['stage1'] = args.step_size_stage1
        if args.step_size_decoder_last is not None:
            step_size['decoder_last'] = args.step_size_decoder_last
        print(f"  step_size (分别设置): {step_size}")
    else:
        print(f"  step_size: 使用配置文件中的值")
    
    results = {}
    
    # 测试 1: 使用 EfficientScan 配置
    print("\n" + "-" * 80)
    print("测试 1: RadioMambaNet (使用 EfficientScan 配置)")
    print("-" * 80)
    print_efficient_scan_config(model_config, use_efficient_scan=True, step_size=step_size)
    try:
        model_new = create_model_from_config(config, use_efficient_scan=True, step_size=step_size)
        
        # 参数量
        params_new = sum(p.numel() for p in model_new.parameters())
        print(f"\n  参数量: {params_new:,} ({params_new/1e6:.2f}M)")
        
        # 性能测试
        if device.type == 'cuda':
            print(f"  开始性能测试 (迭代 {args.iterations} 次，使用 CUDA Events 测量)...")
        else:
            print(f"  开始性能测试 (迭代 {args.iterations} 次)...")
        avg_time_new, peak_memory_new = benchmark_model(
            model_new, input_shape, device, num_iterations=args.iterations
        )
        print(f"  平均推理时间 (GPU): {avg_time_new:.3f} ms")
        print(f"  峰值显存: {peak_memory_new:.2f} MB")
        
        results['new'] = {
            'params': params_new,
            'time': avg_time_new,
            'memory': peak_memory_new
        }
        
        # 清理显存
        del model_new
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        results['new'] = None
    
    # 测试 2: 不使用 EfficientScan (全部使用原始 Mamba)
    print("\n" + "-" * 80)
    print("测试 2: RadioMambaNet (不使用 EfficientScan，全部使用原始 Mamba)")
    print("-" * 80)
    print_efficient_scan_config(model_config, use_efficient_scan=False)
    try:
        model_old = create_model_from_config(config, use_efficient_scan=False)
        
        # 参数量
        params_old = sum(p.numel() for p in model_old.parameters())
        print(f"\n  参数量: {params_old:,} ({params_old/1e6:.2f}M)")
        
        # 性能测试
        if device.type == 'cuda':
            print(f"  开始性能测试 (迭代 {args.iterations} 次，使用 CUDA Events 测量)...")
        else:
            print(f"  开始性能测试 (迭代 {args.iterations} 次)...")
        avg_time_old, peak_memory_old = benchmark_model(
            model_old, input_shape, device, num_iterations=args.iterations
        )
        print(f"  平均推理时间 (GPU): {avg_time_old:.3f} ms")
        print(f"  峰值显存: {peak_memory_old:.2f} MB")
        
        results['old'] = {
            'params': params_old,
            'time': avg_time_old,
            'memory': peak_memory_old
        }
        
        # 清理显存
        del model_old
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        results['old'] = None
    
    # 对比结果
    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)
    
    if results['new'] and results['old']:
        print(f"\n参数量对比:")
        print(f"  EfficientScan 配置: {results['new']['params']:,} ({results['new']['params']/1e6:.2f}M)")
        print(f"  原始 Mamba 配置: {results['old']['params']:,} ({results['old']['params']/1e6:.2f}M)")
        param_diff = results['new']['params'] - results['old']['params']
        param_ratio = results['new']['params'] / results['old']['params']
        print(f"  差异: {param_diff:+,} ({param_ratio:.2f}x)")
        
        print(f"\n推理时间对比:")
        print(f"  EfficientScan 配置: {results['new']['time']:.3f} ms")
        print(f"  原始 Mamba 配置: {results['old']['time']:.3f} ms")
        time_ratio = results['new']['time'] / results['old']['time']
        if time_ratio > 1:
            print(f"  EfficientScan 配置慢 {time_ratio:.2f}x")
        else:
            print(f"  EfficientScan 配置快 {1/time_ratio:.2f}x")
        
        print(f"\n显存使用对比:")
        print(f"  EfficientScan 配置: {results['new']['memory']:.2f} MB")
        print(f"  原始 Mamba 配置: {results['old']['memory']:.2f} MB")
        memory_ratio = results['new']['memory'] / results['old']['memory']
        print(f"  EfficientScan 配置使用 {memory_ratio:.2f}x 显存")
        
        # 计算吞吐量 (FPS)
        fps_new = 1000.0 / results['new']['time'] if results['new']['time'] > 0 else 0
        fps_old = 1000.0 / results['old']['time'] if results['old']['time'] > 0 else 0
        print(f"\n吞吐量对比 (FPS):")
        print(f"  EfficientScan 配置: {fps_new:.2f} FPS")
        print(f"  原始 Mamba 配置: {fps_old:.2f} FPS")
        if fps_new > fps_old:
            print(f"  EfficientScan 配置快 {fps_new/fps_old:.2f}x")
        else:
            print(f"  原始 Mamba 配置快 {fps_old/fps_new:.2f}x")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()