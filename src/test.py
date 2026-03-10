# @Description: Testing script for RadioMambaNet v14. Simplified version with single model configuration.

import os
import argparse
import yaml
import glob
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
import time
from collections import defaultdict

# Import torchmetrics for SSIM and PSNR
try:
    from torchmetrics.functional import (
        structural_similarity_index_measure as functional_ssim,
        peak_signal_noise_ratio as functional_psnr
    )
except ImportError:
    functional_ssim = None
    functional_psnr = None
    print("Warning: torchmetrics not found. SSIM and PSNR calculations will be unavailable.")

# Import local modules
from train import RadioMapSeerDataModule, LightningRadioModel


def save_prediction_image(pred_tensor, save_path):
    pred_np = pred_tensor.squeeze().cpu().numpy()
    if pred_np.dtype != np.uint8:
        pred_np = (np.clip(pred_np, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pred_np, mode='L').save(save_path)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def calculate_metrics_for_pair(pred_tensor, target_tensor):
    """
    计算给定预测张量和目标张量的MSE, NMSE, RMSE, SSIM, PSNR。
    pred_tensor 和 target_tensor 预计都是归一化到 0-1 的 float Tensor。
    形状应为 (B, C, H, W)，其中 C=1。
    """
    metrics = {}

    # Ensure tensors are float32 for metric calculations
    pred_tensor = pred_tensor.float()
    target_tensor = target_tensor.float()

    # Ensure tensors have shape (B, 1, H, W) for metric calculations
    if pred_tensor.ndim == 2:  # (H, W) -> (1, 1, H, W)
        pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)
    elif pred_tensor.ndim == 3:  # (C, H, W) -> (1, C, H, W) -> (1, 1, H, W)
        pred_tensor = pred_tensor.unsqueeze(0)
    if pred_tensor.shape[1] > 1:  # If C > 1, take first channel as it's grayscale
        pred_tensor = pred_tensor[:, :1, :, :]

    if target_tensor.ndim == 2:  # (H, W) -> (1, 1, H, W)
        target_tensor = target_tensor.unsqueeze(0).unsqueeze(0)
    elif target_tensor.ndim == 3:  # (C, H, W) -> (1, C, H, W) -> (1, 1, H, W)
        target_tensor = target_tensor.unsqueeze(0)
    if target_tensor.shape[1] > 1:  # If C > 1, take first channel as it's grayscale
        target_tensor = target_tensor[:, :1, :, :]

    # MSE (Mean Squared Error)
    mse_loss_fn = nn.MSELoss(reduction='mean')
    mse = mse_loss_fn(pred_tensor, target_tensor).item()
    metrics['MSE'] = mse

    # RMSE (Root Mean Squared Error)
    rmse = math.sqrt(mse)
    metrics['RMSE'] = rmse

    # NMSE (Normalized Mean Squared Error)
    # Defined as sum((pred-target)^2) / sum(target^2)
    target_squared_mean = mse_loss_fn(
        target_tensor, torch.zeros_like(target_tensor)).item()
    if target_squared_mean < 1e-9:  # 避免除以零，如果目标值接近零
        if mse < 1e-9:  # 预测值也接近零，则认为是完美匹配
            nmse = 0.0
        else:  # 目标值接近零，但预测值不接近零，说明误差相对于目标值无限大
            nmse = float('inf')
    else:
        nmse = mse / target_squared_mean
    metrics['NMSE'] = nmse

    # SSIM (Structural Similarity Index Measure)
    if functional_ssim:
        ssim_val = functional_ssim(
            pred_tensor, target_tensor, data_range=1.0).item()
        metrics['SSIM'] = ssim_val
    else:
        metrics['SSIM'] = float('nan')  # 如果torchmetrics未安装，则为NaN

    # PSNR (Peak Signal-to-Noise Ratio)
    if functional_psnr:
        psnr_val = functional_psnr(
            pred_tensor, target_tensor, data_range=1.0).item()
        metrics['PSNR'] = psnr_val
    else:
        metrics['PSNR'] = float('nan')  # 如果torchmetrics未安装，则为NaN

    return metrics


def find_best_checkpoint(checkpoint_dir, filename_pattern):
    candidate_checkpoints = glob.glob(os.path.join(
        checkpoint_dir, filename_pattern + "*.ckpt"))
    if not candidate_checkpoints:
        return None

    checkpoints_with_metric = []
    for ckpt in candidate_checkpoints:
        try:
            loss_part = ckpt.split('val_total_loss=')[-1].split('.ckpt')[0]
            metric_val = float(loss_part)
            checkpoints_with_metric.append((metric_val, ckpt))
        except (ValueError, IndexError):
            continue  # Skip files that don't match the loss pattern

    if checkpoints_with_metric:
        checkpoints_with_metric.sort(key=lambda x: x[0])
        return checkpoints_with_metric[0][1]
    else:
        # Fallback to latest modified if no loss value in name
        candidate_checkpoints.sort(key=os.path.getmtime, reverse=True)
        return candidate_checkpoints[0]


def test_model(config_path, specified_checkpoint_path=None, device=None):
    cfg = load_config(config_path)
    testing_cfg = cfg['testing']

    # Build model parameters directly from config (no more multi-scale)
    # 与 train.py 中的参数保持一致，包括 EfficientScan 相关参数
    model_params = {
        'in_channels': cfg['Model']['in_channels'],
        'out_channels': cfg['Model']['out_channels'],
        'dims': cfg['Model']['dims'],
        'depths': cfg['Model']['depths'],
        'ssm_d_state': cfg['Model']['ssm_d_state'],
        'ssm_d_conv': cfg['Model']['ssm_d_conv'],
        'ssm_expand': cfg['Model']['ssm_expand'],
        'efficient_scan_stage0': cfg['Model'].get('efficient_scan_stage0', True),
        'step_size_stage0': cfg['Model'].get('step_size_stage0', 2),
        'efficient_scan_stage1': cfg['Model'].get('efficient_scan_stage1', True),
        'step_size_stage1': cfg['Model'].get('step_size_stage1', 2),
        'efficient_scan_decoder_last': cfg['Model'].get('efficient_scan_decoder_last', True),
        'step_size_decoder_last': cfg['Model'].get('step_size_decoder_last', 2)
    }

    # Use training config directly
    training_config = cfg['training'].copy()

    output_dir = testing_cfg['results_save_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Prediction images will be saved to: {output_dir}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = specified_checkpoint_path or testing_cfg.get(
        'checkpoint_path')
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(
            f"Warning: Checkpoint path '{checkpoint_path}' not found. Attempting to find best checkpoint automatically.")
        checkpoint_dir = cfg['callbacks']['checkpoint_best']['dirpath']
        filename_base = cfg['callbacks']['checkpoint_best']['filename'].split('{')[
            0]
        checkpoint_path = find_best_checkpoint(checkpoint_dir, filename_base)
        if not checkpoint_path:
            raise FileNotFoundError(
                f"Could not find any checkpoint in {checkpoint_dir}. Please specify a valid path in config.")

    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # 严格按照配置文件中的参数加载模型，不进行自动检测和调整
    print(f"\n使用配置文件中的模型参数:")
    print(f"  efficient_scan_stage0: {model_params['efficient_scan_stage0']}")
    print(f"  efficient_scan_stage1: {model_params['efficient_scan_stage1']}")
    print(f"  efficient_scan_decoder_last: {model_params['efficient_scan_decoder_last']}")
    print(f"  step_size_stage0: {model_params['step_size_stage0']}")
    print(f"  step_size_stage1: {model_params['step_size_stage1']}")
    print(f"  step_size_decoder_last: {model_params['step_size_decoder_last']}")

    # Load model with correct parameters from v14 config
    model = LightningRadioModel.load_from_checkpoint(
        checkpoint_path,
        model_params=model_params,
        training_config=training_config,
        strict=False  # 允许部分权重不匹配，但会打印警告
    )
    model.to(device)
    model.eval()

    # Prepare test data config
    test_data_config = cfg['data'].copy()
    test_data_config['batch_size'] = testing_cfg['test_batch_size']
    
    # 获取数据集类型（从配置中读取，默认 'radiomamba'）
    dataset_type = test_data_config.get('dataset_type', 'radiomamba')
    print(f"Using dataset type: {dataset_type}")

    data_module = RadioMapSeerDataModule(
        data_config=test_data_config, 
        seed=cfg.get('seed', 42),
        dataset_type=dataset_type)
    data_module.setup(stage='test')
    test_dataloader = data_module.test_dataloader()

    num_save_images = testing_cfg.get('num_save_images', 4000)
    saved_count = 0
    total_processing_time = 0.0  # 总处理时间
    total_inference_time = 0.0  # 纯推理时间
    
    # 用于累积指标
    total_metrics = defaultdict(float)

    print(f"Test dataloader length: {len(test_dataloader)}")
    print(f"Test dataset length: {len(data_module.test_dataset)}")

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Generating Predictions"):
            if saved_count >= num_save_images:
                break

            # 开始计算总处理时间
            batch_start_time = time.time()

            # 现在batch_size=1，所以每个batch只有一张图片
            inputs, targets, names = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 单独计算推理时间
            inference_start_time = time.time()
            predictions = model(inputs)
            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time
            total_inference_time += inference_time

            # 与训练时 validation_step 保持一致：clamp 到 [0, 1]
            predictions_clamped = torch.clamp(predictions, 0.0, 1.0)

            # 计算指标（使用第一张图片，因为batch_size=1）
            pred_tensor = predictions_clamped[0:1]  # 保持batch维度，使用clamp后的结果
            target_tensor = targets[0:1]    # 保持batch维度
            batch_metrics = calculate_metrics_for_pair(pred_tensor, target_tensor)
            
            # 累积指标
            for k, v in batch_metrics.items():
                if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    total_metrics[k] += v

            # 由于batch_size=1，直接处理第一张（也是唯一一张）图片
            # 使用 clamp 后的结果保存，与训练时 validation_step 保持一致
            image_name = names[0]
            # 确保文件名有 .png 扩展名
            if not image_name.endswith('.png'):
                image_name = image_name + '.png'
            save_full_path = os.path.join(output_dir, image_name)
            save_prediction_image(predictions_clamped[0], save_full_path)
            saved_count += 1

            # 结束计算总处理时间
            batch_end_time = time.time()
            batch_processing_time = batch_end_time - batch_start_time
            total_processing_time += batch_processing_time

            # 每500张图片打印一次进度
            if saved_count % 500 == 0:
                avg_metrics_str = ", ".join([
                    f"{k}: {total_metrics[k]/saved_count:.4f}" 
                    for k in ['MSE', 'RMSE', 'NMSE', 'SSIM', 'PSNR'] 
                    if k in total_metrics
                ])
                print(
                    f"Processed {saved_count} images, current file: {image_name}")
                if avg_metrics_str:
                    print(f"  Current averages: {avg_metrics_str}")

    # 计算平均时间和指标
    if saved_count > 0:
        avg_processing_time = total_processing_time / saved_count
        avg_inference_time = total_inference_time / saved_count
        avg_io_time = avg_processing_time - avg_inference_time
        
        # 计算平均指标
        avg_metrics = {}
        for k, v in total_metrics.items():
            avg_metrics[k] = v / saved_count
    else:
        avg_processing_time = avg_inference_time = avg_io_time = 0.0
        avg_metrics = {}

    print(f"\n--- Prediction Complete ---")
    print(f"Saved {saved_count} prediction images to {output_dir}")
    print(f"\n--- Timing Information ---")
    print(f"Total processing time: {total_processing_time:.8f} seconds")
    print(f"Total inference time: {total_inference_time:.8f} seconds")
    print(f"Average total processing time per image: {avg_processing_time:.8f} seconds")
    print(f"Average inference time per image: {avg_inference_time:.8f} seconds")
    print(f"Average I/O time per image: {avg_io_time:.8f} seconds")
    
    print(f"\n--- Average Metrics (over {saved_count} images) ---")
    metric_order = ['MSE', 'RMSE', 'NMSE', 'SSIM', 'PSNR']
    for metric_name in metric_order:
        if metric_name in avg_metrics:
            value = avg_metrics[metric_name]
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                print(f"{metric_name}: {value}")
            else:
                print(f"{metric_name}: {value:.6f}")

    # 保存详细的结果到文件
    results_file_path = os.path.join(output_dir, 'test_results.txt')
    with open(results_file_path, 'w') as f:
        f.write(f"# Test Results\n")
        f.write(f"Total images processed: {saved_count}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"\n--- Timing Information (seconds) ---\n")
        f.write(f"Total processing time: {total_processing_time:.8f}\n")
        f.write(f"Total inference time: {total_inference_time:.8f}\n")
        f.write(f"Average total processing time: {avg_processing_time:.8f}\n")
        f.write(f"Average inference time: {avg_inference_time:.8f}\n")
        f.write(f"Average I/O time: {avg_io_time:.8f}\n")
        
        f.write(f"\n--- Average Metrics ---\n")
        for metric_name in metric_order:
            if metric_name in avg_metrics:
                value = avg_metrics[metric_name]
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    f.write(f"{metric_name}: {value}\n")
                else:
                    f.write(f"{metric_name}: {value:.6f}\n")

    print(f"\nDetailed results saved to: {results_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test RadioMambaNet v14 model.")
    parser.add_argument('--config', type=str, default='../configs/config_withcars.yaml',
                        help='Path to the YAML config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='(Optional) Path to a specific model checkpoint file.')
    args = parser.parse_args()

    # 设置测试设备为 cuda:0
    device = torch.device("cuda:3")
    test_model(args.config, args.checkpoint, device)
