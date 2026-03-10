#!/usr/bin/env python3
"""
批量测试不同 stage0 和 stage1 step_size 组合的性能
固定 decoder_last 的 step_size 为 2

使用示例:

1. 只测试 stage0，固定 stage1=2:
python benchmark_stage0_sweep.py \
    --config ../configs/config_nocars_512_effcient.yaml \
    --input_size 1024 \
    --step_size_stage0_list 2 4 \
    --include_baseline

2. 测试 stage0 和 stage1 的所有组合 (2×2=4 种组合):
python benchmark_stage0_sweep.py \
    --config ../configs/config_nocars_512_effcient.yaml \
    --input_size 1024 \
    --step_size_stage0_list 2 4 \
    --step_size_stage1_list 2 4 \
    --include_baseline \
    --output_csv results_stage0_stage1_combinations_1024.csv

3. 测试 stage0、stage1 和 decoder_last 的所有组合 (2×2×2=8 种组合):

python benchmark_stage0_sweep.py \
    --config ../configs/config_nocars_512_effcient.yaml \
    --input_size 512 \
    --step_size_stage0_list 2 4 \
    --step_size_stage1_list 2 4 \
    --step_size_decoder_last_list 2 4 \
    --include_baseline \
    --output_csv results_all_combinations_512.csv


python benchmark_stage0_sweep.py \
    --config ../configs/config_nocars_1024_effcient.yaml \
    --input_size 1024 \
    --step_size_stage0_list 2 4 \
    --step_size_stage1_list 2 4 \
    --step_size_decoder_last_list 2 4 \
    --include_baseline \
    --output_csv results_all_combinations_1024.csv



python benchmark_stage0_sweep.py \
    --config ../configs/config_nocars_1024_effcient.yaml \
    --input_size 256 \
    --step_size_stage0_list 2 4 \
    --step_size_stage1_list 2 4 \
    --step_size_decoder_last_list 2 4 \
    --include_baseline \
    --output_csv results_all_combinations_1024.csv
"""

import torch
import torch.nn as nn
import time
import sys
import os
import yaml
import argparse
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
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
        base_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        activation_memory = peak_memory - base_memory
        
        # 计算模型参数显存
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2  # MB
    else:
        peak_memory = 0
        activation_memory = 0
        param_memory = 0
        base_memory = 0
    
    return avg_time, peak_memory, activation_memory, param_memory


def load_config(config_path):
    """从 YAML 配置文件加载参数"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config, use_efficient_scan=True, step_size_stage0=2, step_size_stage1=2, step_size_decoder_last=2):
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
        efficient_scan_stage1 = model_config.get('efficient_scan_stage1', True)
        efficient_scan_decoder_last = model_config.get('efficient_scan_decoder_last', True)
    else:
        efficient_scan_stage0 = False
        efficient_scan_stage1 = False
        efficient_scan_decoder_last = False
        step_size_stage0 = 1
        step_size_stage1 = 1
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


def test_single_config(config, input_size, device, step_size_stage0, step_size_stage1=2, step_size_decoder_last=2, 
                       iterations=50, baseline=False):
    """测试单个配置"""
    model_config = config.get('Model', config.get('model', {}))
    in_channels = model_config.get('in_channels', 3)
    input_shape = (1, in_channels, input_size, input_size)
    
    try:
        if baseline:
            model = create_model_from_config(config, use_efficient_scan=False)
        else:
            model = create_model_from_config(
                config, 
                use_efficient_scan=True,
                step_size_stage0=step_size_stage0,
                step_size_stage1=step_size_stage1,
                step_size_decoder_last=step_size_decoder_last
            )
        
        # 参数量
        params = sum(p.numel() for p in model.parameters())
        
        # 性能测试
        avg_time, peak_memory, activation_memory, param_memory = benchmark_model(
            model, input_shape, device, num_iterations=iterations
        )
        
        # 计算吞吐量
        fps = 1000.0 / avg_time if avg_time > 0 else 0
        
        # 清理显存
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return {
            'params': params,
            'params_M': params / 1e6,
            'time_ms': avg_time,
            'fps': fps,
            'peak_memory_MB': peak_memory,
            'activation_memory_MB': activation_memory,
            'param_memory_MB': param_memory,
            'success': True
        }
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return {
            'params': 0,
            'params_M': 0,
            'time_ms': 0,
            'fps': 0,
            'peak_memory_MB': 0,
            'activation_memory_MB': 0,
            'param_memory_MB': 0,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='批量测试不同 stage0 和 stage1 step_size 组合的性能')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--input_size', type=int, default=1024,
                       help='输入图像尺寸 (默认: 1024)')
    parser.add_argument('--iterations', type=int, default=50,
                       help='测试迭代次数 (默认: 50)')
    parser.add_argument('--step_size_stage0_list', type=int, nargs='+', 
                       default=[1, 2, 3, 4, 5, 6],
                       help='要测试的 stage0 step_size 列表 (默认: 1 2 3 4 5 6)')
    parser.add_argument('--step_size_stage1_list', type=int, nargs='+', default=None,
                       help='要测试的 stage1 step_size 列表，如果提供则测试所有组合 (默认: 使用 --step_size_stage1)')
    parser.add_argument('--step_size_stage1', type=int, default=2,
                       help='固定的 stage1 step_size，当 --step_size_stage1_list 未提供时使用 (默认: 2)')
    parser.add_argument('--step_size_decoder_last_list', type=int, nargs='+', default=None,
                       help='要测试的 decoder_last step_size 列表，如果提供则测试所有组合 (默认: 使用 --step_size_decoder_last)')
    parser.add_argument('--step_size_decoder_last', type=int, default=2,
                       help='固定的 decoder_last step_size，当 --step_size_decoder_last_list 未提供时使用 (默认: 2)')
    parser.add_argument('--output_csv', type=str, default=None,
                       help='输出 CSV 文件路径 (默认: 自动生成)')
    parser.add_argument('--include_baseline', action='store_true',
                       help='包含 baseline (不使用 EfficientScan) 的测试')
    args = parser.parse_args()
    
    # 确定 stage1 和 decoder_last 的测试列表
    if args.step_size_stage1_list is not None:
        stage1_list = args.step_size_stage1_list
        test_stage1_combinations = True
    else:
        stage1_list = [args.step_size_stage1]
        test_stage1_combinations = False
    
    if args.step_size_decoder_last_list is not None:
        decoder_last_list = args.step_size_decoder_last_list
        test_decoder_combinations = True
    else:
        decoder_last_list = [args.step_size_decoder_last]
        test_decoder_combinations = False
    
    # 判断是否测试组合
    test_combinations = test_stage1_combinations or test_decoder_combinations
    
    print("=" * 80)
    if test_combinations:
        if test_stage1_combinations and test_decoder_combinations:
            print("批量测试不同 stage0、stage1 和 decoder_last step_size 组合的性能")
        elif test_stage1_combinations:
            print("批量测试不同 stage0 和 stage1 step_size 组合的性能")
        else:
            print("批量测试不同 stage0 和 decoder_last step_size 组合的性能")
    else:
        print("批量测试不同 stage0 step_size 的性能")
    print("=" * 80)
    print(f"配置文件: {args.config}")
    print(f"输入尺寸: {args.input_size}×{args.input_size}")
    if test_combinations:
        print(f"测试组合:")
        print(f"  stage0: {args.step_size_stage0_list}")
        if test_stage1_combinations:
            print(f"  stage1: {stage1_list}")
        else:
            print(f"  stage1: {args.step_size_stage1} (固定)")
        if test_decoder_combinations:
            print(f"  decoder_last: {decoder_last_list}")
        else:
            print(f"  decoder_last: {args.step_size_decoder_last} (固定)")
        total_combinations = len(args.step_size_stage0_list) * len(stage1_list) * len(decoder_last_list)
        print(f"  总组合数: {total_combinations}")
    else:
        print(f"固定配置: stage1={args.step_size_stage1}, decoder_last={args.step_size_decoder_last}")
        print(f"测试 stage0 step_size: {args.step_size_stage0_list}")
    print("=" * 80)
    
    # 加载配置
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        return
    
    config = load_config(args.config)
    model_config = config.get('Model', config.get('model', {}))
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    
    # 存储结果
    results = []
    
    # 测试 baseline (如果启用)
    if args.include_baseline:
        print("\n" + "-" * 80)
        print("测试 Baseline (不使用 EfficientScan)")
        print("-" * 80)
        baseline_result = test_single_config(
            config, args.input_size, device, 
            step_size_stage0=1, 
            step_size_stage1=1, 
            step_size_decoder_last=1,
            iterations=args.iterations,
            baseline=True
        )
        baseline_result['step_size_stage0'] = 'Baseline'
        baseline_result['step_size_stage1'] = 'N/A'
        baseline_result['step_size_decoder_last'] = 'N/A'
        results.append(baseline_result)
        
        if baseline_result['success']:
            print(f"  参数量: {baseline_result['params']:,} ({baseline_result['params_M']:.2f}M)")
            print(f"  推理时间: {baseline_result['time_ms']:.3f} ms")
            print(f"  吞吐量: {baseline_result['fps']:.2f} FPS")
            print(f"  峰值显存: {baseline_result['peak_memory_MB']:.2f} MB")
            print(f"  激活值显存: {baseline_result['activation_memory_MB']:.2f} MB")
    
    # 测试不同的 stage0、stage1 和 decoder_last 组合
    if test_combinations:
        # 测试所有组合（三层嵌套循环）
        for step_size_stage0 in args.step_size_stage0_list:
            for step_size_stage1 in stage1_list:
                for step_size_decoder_last in decoder_last_list:
                    print("\n" + "-" * 80)
                    print(f"测试组合: stage0={step_size_stage0}, stage1={step_size_stage1}, decoder_last={step_size_decoder_last}")
                    print("-" * 80)
                    
                    result = test_single_config(
                        config, args.input_size, device,
                        step_size_stage0=step_size_stage0,
                        step_size_stage1=step_size_stage1,
                        step_size_decoder_last=step_size_decoder_last,
                        iterations=args.iterations,
                        baseline=False
                    )
                    
                    result['step_size_stage0'] = step_size_stage0
                    result['step_size_stage1'] = step_size_stage1
                    result['step_size_decoder_last'] = step_size_decoder_last
                    results.append(result)
                    
                    if result['success']:
                        print(f"  参数量: {result['params']:,} ({result['params_M']:.2f}M)")
                        print(f"  推理时间: {result['time_ms']:.3f} ms")
                        print(f"  吞吐量: {result['fps']:.2f} FPS")
                        print(f"  峰值显存: {result['peak_memory_MB']:.2f} MB")
                        print(f"  激活值显存: {result['activation_memory_MB']:.2f} MB")
                    else:
                        print(f"  ❌ 测试失败: {result.get('error', 'Unknown error')}")
    else:
        # 只测试 stage0，stage1 和 decoder_last 固定
        for step_size_stage0 in args.step_size_stage0_list:
            print("\n" + "-" * 80)
            print(f"测试 stage0 step_size = {step_size_stage0}")
            print(f"  (stage1={args.step_size_stage1}, decoder_last={args.step_size_decoder_last})")
            print("-" * 80)
            
            result = test_single_config(
                config, args.input_size, device,
                step_size_stage0=step_size_stage0,
                step_size_stage1=args.step_size_stage1,
                step_size_decoder_last=args.step_size_decoder_last,
                iterations=args.iterations,
                baseline=False
            )
            
            result['step_size_stage0'] = step_size_stage0
            result['step_size_stage1'] = args.step_size_stage1
            result['step_size_decoder_last'] = args.step_size_decoder_last
            results.append(result)
            
            if result['success']:
                print(f"  参数量: {result['params']:,} ({result['params_M']:.2f}M)")
                print(f"  推理时间: {result['time_ms']:.3f} ms")
                print(f"  吞吐量: {result['fps']:.2f} FPS")
                print(f"  峰值显存: {result['peak_memory_MB']:.2f} MB")
                print(f"  激活值显存: {result['activation_memory_MB']:.2f} MB")
            else:
                print(f"  ❌ 测试失败: {result.get('error', 'Unknown error')}")
    
    # 输出结果表格
    print("\n" + "=" * 80)
    print("结果汇总")
    print("=" * 80)
    
    # 表格头部
    header = f"{'Stage0':<8} {'Stage1':<8} {'Decoder':<8} {'参数量(M)':<12} {'时间(ms)':<12} {'FPS':<10} {'峰值显存(MB)':<15} {'激活值显存(MB)':<18}"
    print(header)
    print("-" * 100)
    
    # 计算相对于 baseline 的提升（如果有 baseline）
    baseline_time = None
    baseline_memory = None
    if args.include_baseline and results[0]['success']:
        baseline_time = results[0]['time_ms']
        baseline_memory = results[0]['peak_memory_MB']
    
    for result in results:
        if result['success']:
            stage0 = str(result['step_size_stage0'])
            stage1 = str(result['step_size_stage1'])
            decoder = str(result['step_size_decoder_last'])
            params = f"{result['params_M']:.2f}"
            time_ms = f"{result['time_ms']:.3f}"
            fps = f"{result['fps']:.2f}"
            peak_mem = f"{result['peak_memory_MB']:.2f}"
            act_mem = f"{result['activation_memory_MB']:.2f}"
            
            # 计算速度提升
            if baseline_time and result['step_size_stage0'] != 'Baseline':
                speedup = baseline_time / result['time_ms']
                time_ms += f" ({speedup:.2f}x)"
            
            # 计算显存变化
            if baseline_memory and result['step_size_stage0'] != 'Baseline':
                mem_ratio = result['peak_memory_MB'] / baseline_memory
                peak_mem += f" ({mem_ratio:.2f}x)"
            
            row = f"{stage0:<8} {stage1:<8} {decoder:<8} {params:<12} {time_ms:<20} {fps:<10} {peak_mem:<20} {act_mem:<18}"
            print(row)
        else:
            print(f"{result['step_size_stage0']:<8} {'ERROR':<8} {'ERROR':<8} {'-':<12} {'-':<12} {'-':<10} {'-':<15} {'-':<18}")
    
    # 保存到 CSV
    if args.output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = f"benchmark_stage0_sweep_{args.input_size}_{timestamp}.csv"
    
    print(f"\n保存结果到: {args.output_csv}")
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'step_size_stage0', 'step_size_stage1', 'step_size_decoder_last',
            'params', 'params_M', 'time_ms', 'fps', 
            'peak_memory_MB', 'activation_memory_MB', 'param_memory_MB', 'success'
        ])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

