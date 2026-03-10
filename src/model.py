# @Description:
#   Version 13: Simplified version based on v12 Radio-MambaNet architecture.
#   Implements a hybrid U-Net architecture combining a new SS2D Mamba block for global context
#   and a residual convolution block for local features within each stage.
#   Removed multi-scale configuration support for simplified single-model design.

import os
import ctypes
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

# --- Setup LD_LIBRARY_PATH for selective_scan_cuda_core ---
# 在导入 selective_scan_cuda_core 之前设置库路径
def setup_torch_lib_path():
    """设置 PyTorch 库路径，确保 selective_scan_cuda_core 可以找到 libc10.so"""
    try:
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        libc10_path = os.path.join(torch_lib_path, 'libc10.so')
        
        # 设置环境变量（虽然对当前进程可能无效，但可以用于子进程）
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if torch_lib_path not in current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}"
        
        # 尝试手动加载 libc10.so（如果存在）
        if os.path.exists(libc10_path):
            try:
                ctypes.CDLL(libc10_path, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                # 如果加载失败，不影响程序运行
                pass
    except Exception:
        # 如果设置失败，不影响程序运行
        pass

# 执行设置
setup_torch_lib_path()

# 尝试导入 selective_scan_cuda_core（参考 test1.py 的导入方式）
try:
    import selective_scan_cuda_core
    SELECTIVE_SCAN_AVAILABLE = True
    print("✓ Successfully imported selective_scan_cuda_core")
except ImportError as e:
    selective_scan_cuda_core = None
    SELECTIVE_SCAN_AVAILABLE = False
    print(f"⚠ Could not import selective_scan_cuda_core: {e}")
    # 如果导入失败，可能是 LD_LIBRARY_PATH 未设置或模块未安装
    import warnings
    if "libc10.so" in str(e) or "cannot open shared object file" in str(e):
        warnings.warn(
            "selective_scan_cuda_core 导入失败，可能是缺少 libc10.so。\n"
            "请在启动 Python 前设置环境变量：\n"
            "export LD_LIBRARY_PATH=$(python -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))'):$LD_LIBRARY_PATH\n"
            "或者使用：\n"
            "export LD_LIBRARY_PATH=/home/zqm1/anaconda3/envs/Mamba_py310/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH",
            ImportWarning
        )

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("Warning: `mamba_ssm` package not found. Please install with `pip install mamba-ssm`.")

# --- 添加 EfficientVMamba 路径并导入 SS2D ---
import sys
efficient_vmamba_path = "/home/zqm1/workspace2/EfficientVMamba-main/classification"
if efficient_vmamba_path not in sys.path:
    sys.path.insert(0, efficient_vmamba_path)
    sys.path.insert(0, os.path.join(efficient_vmamba_path, 'models'))

try:
    from models.vmamba_efficient import SS2D
    SS2D_AVAILABLE = True
    print("✓ Successfully imported SS2D from EfficientVMamba")
except ImportError as e:
    SS2D = None
    SS2D_AVAILABLE = False
    print(f"⚠ Could not import SS2D: {e}")

# --- Helper Modules ---

import math







class ResidualConvBlock(nn.Module):
    """
    A standard residual convolutional block for capturing local features.
    Conv -> Norm -> Act -> Conv -> Norm
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1,
                      groups=dim, bias=False),  # Depthwise
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),  # Pointwise
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_branch(x)


class SS2D_Mamba(nn.Module):
    """
    2D Visual Mamba Block with conditional SS2D or Mamba usage.
    根据 use_efficient_scan 参数选择使用 EfficientVMamba 的 SS2D 或原始 Mamba。
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, 
                 expand: int = 2, step_size: int = 2, use_efficient_scan: bool = False):
        super().__init__()
        
        self.dim = dim
        self.use_efficient_scan = use_efficient_scan
        self.step_size = step_size
        
        if use_efficient_scan:
            # 使用 EfficientVMamba 的 SS2D
            if not SS2D_AVAILABLE:
                raise ImportError(
                    "SS2D from EfficientVMamba is not available. "
                    "Please ensure EfficientVMamba is properly installed.")
            
            # 创建 SS2D 实例
            # 重要：d_model 必须等于输入 x_channel_last 的最后一个维度（通道数 C）
            self.ss2d = SS2D(
                d_model=dim,  # 必须等于输入 x_channel_last 的最后一个维度（通道数 C）
                d_state=d_state,
                ssm_ratio=float(expand),  # expand 对应 ssm_ratio
                ssm_rank_ratio=float(expand),  # 通常与 ssm_ratio 相同
                dt_rank="auto",
                act_layer=nn.SiLU,
                d_conv=1,
                conv_bias=True,
                dropout=0.0,
                bias=False,
                forward_type="v2",  # 使用 v2 版本（支持 EfficientScan）
                step_size=self.step_size,
            )
            self.mamba = None
            self.norm = None
        else:
            # 使用原始 Mamba（双向扫描）
            if Mamba is None:
                raise ImportError(
                    "Mamba-ssm is not installed. Please install it to use SS2D_Mamba.")
            
            self.norm = nn.LayerNorm(dim)
            self.mamba = Mamba(d_model=dim, d_state=d_state,
                              d_conv=d_conv, expand=expand)
            self.ss2d = None

    def forward(self, x: torch.Tensor):
        """
        x: input tensor of shape (B, C, H, W) - channel first
        Returns: output tensor of shape (B, C, H, W) - channel first
        """
        if self.use_efficient_scan:
            # 使用 SS2D（EfficientVMamba）
            B, C, H, W = x.shape
            
            # SS2D 期望输入格式为 (B, H, W, C) - channel last
            # 转换格式：(B, C, H, W) -> (B, H, W, C)
            x_channel_last = x.permute(0, 2, 3, 1).contiguous()
            
            # 调用 SS2D
            out_channel_last = self.ss2d(x_channel_last)
            
            # 转换回 channel first 格式：(B, H, W, C) -> (B, C, H, W)
            out = out_channel_last.permute(0, 3, 1, 2).contiguous()
            
            return out
        else:
            # 使用原始 Mamba（双向扫描）
            B, C, H, W = x.shape
            
            # LayerNorm expects (B, ..., C)
            x_norm = x.permute(0, 2, 3, 1).contiguous()
            x_norm = self.norm(x_norm).permute(0, 3, 1, 2).contiguous()
            
            # ========== 双向Mamba扫描路径 ==========
            # Reshape for sequence processing
            x_seq = x_norm.view(B, C, H * W).transpose(1, 2).contiguous()  # (B, L, C)

            # Forward scan
            out_fwd = self.mamba(x_seq)

            # Backward scan
            out_bwd = self.mamba(x_seq.flip(dims=[1])).flip(dims=[1])

            # Combine and reshape back to 2D
            x_reconstructed = (out_fwd + out_bwd).transpose(1, 2).contiguous().view(B, C, H, W)
            
            return x_reconstructed


class MambaConvBlock(nn.Module):
    """
    The core hybrid block of Radio-MambaNet.
    It contains two parallel branches:
    1. A Mamba branch (SS2D_Mamba) for global, long-range context.
    2. A Convolutional branch (ResidualConvBlock) for local, detailed features.
    """

    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int,
                 step_size: int = 1, use_efficient_scan: bool = False):
        super().__init__()
        self.mamba_branch = SS2D_Mamba(dim, d_state, d_conv, expand,
                                       step_size=step_size, 
                                       use_efficient_scan=use_efficient_scan)
        self.conv_branch = ResidualConvBlock(dim)

    def forward(self, x):
        # The input 'x' is fed into both branches simultaneously
        x_mamba = self.mamba_branch(x)
        x_conv = self.conv_branch(x)
        # The outputs are fused by element-wise addition
        return x_mamba + x_conv


# --- Main Model: RadioMambaNet ---

class RadioMambaNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1,
                 dims: List[int] = [32, 64, 128, 256],
                 depths: List[int] = [2, 2, 2, 2],
                 ssm_d_state: int = 16,
                 ssm_d_conv: int = 4,
                 ssm_expand: int = 2,
                 efficient_scan_stage0: bool = True,  # 是否在Stage 0使用EfficientScan
                 step_size_stage0: int = 2,  # Stage 0的step_size
                 efficient_scan_stage1: bool = True,  # 是否在Stage 1使用EfficientScan
                 step_size_stage1: int = 2,  # Stage 1的step_size
                 efficient_scan_decoder_last: bool = True,  # 是否在解码器最后一个阶段使用EfficientScan
                 step_size_decoder_last: int = 2):  # 解码器最后一个阶段的step_size
        super().__init__()
        self.out_channels = out_channels

        # --- Input Projection ---
        self.patch_embed = nn.Conv2d(
            in_channels, dims[0], kernel_size=3, stride=1, padding=1)

        # --- Encoder ---
        self.encoder_stages = nn.ModuleList()
        for i in range(len(dims)):
            # Stage 0和Stage 1可以使用EfficientScan
            if i == 0:
                use_efficient_scan = efficient_scan_stage0
                step_size = step_size_stage0 if use_efficient_scan else 1
            elif i == 1:
                use_efficient_scan = efficient_scan_stage1
                step_size = step_size_stage1 if use_efficient_scan else 1
            else:
                use_efficient_scan = False
                step_size = 1
            
            stage_blocks = nn.ModuleList([
                MambaConvBlock(dims[i], d_state=ssm_d_state,
                              d_conv=ssm_d_conv, expand=ssm_expand,
                              step_size=step_size,
                              use_efficient_scan=use_efficient_scan)
                for _ in range(depths[i])
            ])
            self.encoder_stages.append(stage_blocks)

            # Downsampling layer for all but the last stage
            if i < len(dims) - 1:
                downsample = nn.Conv2d(
                    dims[i], dims[i+1], kernel_size=2, stride=2)
                self.encoder_stages.append(downsample)

        # --- Bottleneck ---
        # The last stage of the encoder acts as the bottleneck
        bottleneck_dim = dims[-1]
        self.bottleneck = nn.Sequential(*[
            MambaConvBlock(bottleneck_dim, d_state=ssm_d_state,
                          d_conv=ssm_d_conv, expand=ssm_expand,
                          step_size=1,  # Bottleneck不使用EfficientScan
                          use_efficient_scan=False)
            for _ in range(depths[-1])
        ])

        # --- Decoder ---
        self.decoder_stages = nn.ModuleList()
        reversed_dims = dims[::-1]  # [256, 128, 64, 32]

        for i in range(len(reversed_dims) - 1):
            # Upsampling layer
            upsample = nn.ConvTranspose2d(
                reversed_dims[i], reversed_dims[i+1], kernel_size=2, stride=2)
            self.decoder_stages.append(upsample)

            # 判断是否是解码器的最后一个阶段（从128×128到256×256）
            # i == len(reversed_dims) - 2 对应最后一个decoder stage
            is_last_decoder_stage = (i == len(reversed_dims) - 2)
            
            # 最后一个解码器阶段可以使用EfficientScan
            if is_last_decoder_stage:
                use_efficient_scan = efficient_scan_decoder_last
                step_size = step_size_decoder_last if use_efficient_scan else 1
            else:
                use_efficient_scan = False
                step_size = 1

            # MambaConv blocks for feature fusion and refinement
            # The input to this block will be concatenated features (skip + upsampled)
            decoder_conv_dim = reversed_dims[i+1] * 2
            fusion_conv = nn.Conv2d(
                decoder_conv_dim, reversed_dims[i+1], kernel_size=1)
            decoder_blocks = nn.ModuleList([
                MambaConvBlock(
                    reversed_dims[i+1], d_state=ssm_d_state, 
                    d_conv=ssm_d_conv, expand=ssm_expand,
                    step_size=step_size,
                    use_efficient_scan=use_efficient_scan)
                # Match encoder depth
                for _ in range(depths[len(dims) - 2 - i])
            ])
            self.decoder_stages.append(
                nn.Sequential(fusion_conv, *decoder_blocks))

        # --- Final Output Layer ---
        self.final_conv = nn.Sequential(
            nn.Conv2d(dims[0], out_channels, kernel_size=1),
            nn.Sigmoid()  # 将输出约束到 [0, 1] 范围
        )

    def forward(self, x):
        # Input projection
        x = self.patch_embed(x)

        # Encoder path
        skip_connections = []
        for i in range(len(self.encoder_stages)):
            module = self.encoder_stages[i]
            if isinstance(module, nn.ModuleList):  # This is a stage of MambaConvBlocks
                # Save the output of the stage as a skip connection *before* downsampling
                skip_connections.append(x)
                for block in module:
                    x = block(x)
            else:  # This is a downsampling layer
                x = module(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        skip_connections.pop()  # The last skip is from the bottleneck level, which we start from
        for i in range(0, len(self.decoder_stages), 2):
            upsample = self.decoder_stages[i]
            decoder_blocks = self.decoder_stages[i+1]

            x = upsample(x)
            skip = skip_connections.pop()

            # Ensure spatial dimensions match before concatenation
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = decoder_blocks(x)

        # Final output projection
        return self.final_conv(x)


# Alias for compatibility with training script
Model = RadioMambaNet


if __name__ == "__main__":
    import yaml
    import os

    def count_parameters(model):
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_model_size(model):
        """计算模型大小（MB）"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def format_params(params):
        """格式化参数数量为友好的显示格式"""
        if params >= 1e9:
            return f"{params/1e9:.2f}B"
        elif params >= 1e6:
            return f"{params/1e6:.2f}M"
        elif params >= 1e3:
            return f"{params/1e3:.2f}K"
        else:
            return str(params)

    # 加载配置文件 (v14)
    config_path = "/mnt/mydisk/hgjia/scr/RadioMambaUnet/config_v14.yaml"

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 获取模型配置 (v14 simplified)
        model_config = config['Model']
        
        print("=" * 80)
        print("RadioMambaNet v14 模型参数统计")
        print("=" * 80)
        
        print(f"基本配置 - 输入通道: {model_config['in_channels']}, 输出通道: {model_config['out_channels']}")
        print(f"描述: {model_config['description']}")
        
        print(f"\n模型配置参数:")
        print(f"  - 维度: {model_config['dims']}")
        print(f"  - 深度: {model_config['depths']}")
        print(f"  - SSM d_state: {model_config['ssm_d_state']}")
        print(f"  - SSM d_conv: {model_config['ssm_d_conv']}")
        print(f"  - SSM expand: {model_config['ssm_expand']}")
        print(f"  - 批次大小: {config['data']['batch_size']}")
        
        try:
            # 创建模型实例
            model = RadioMambaNet(
                in_channels=model_config['in_channels'],
                out_channels=model_config['out_channels'],
                dims=model_config['dims'],
                depths=model_config['depths'],
                ssm_d_state=model_config['ssm_d_state'],
                ssm_d_conv=model_config['ssm_d_conv'],
                ssm_expand=model_config['ssm_expand']
            )

            # 计算参数数量
            total_params = count_parameters(model)
            model_size_mb = get_model_size(model)
            
            print(f"\n实际参数统计:")
            print(f"  - 总参数数量: {total_params:,} ({format_params(total_params)})")
            print(f"  - 模型大小: {model_size_mb:.2f} MB")
            print(f"  - 配置文件预期: {config['model_info']['params']}")
            print(f"  - 描述: {config['model_info']['description']}")
            
            # 验证参数数量是否与配置文件一致
            expected_params = config['model_info']['params']
            if expected_params.endswith('M'):
                expected_num = float(expected_params[:-1]) * 1e6
                if abs(total_params - expected_num) / expected_num < 0.1:  # 10% 误差范围
                    print(f"  ✅ 参数数量与配置文件一致")
                else:
                    print(f"  ⚠️  参数数量与配置文件不一致 (预期: {expected_params})")
            
        except Exception as e:
            if "mamba_ssm" in str(e).lower():
                print(f"  ❌ 需要安装 mamba-ssm 包: pip install mamba-ssm")
            else:
                print(f"  ❌ 模型创建失败: {str(e)}")
        
        # 详细测试模型
        print("\n" + "=" * 80)
        print("详细测试模型")
        print("=" * 80)
        
        try:
            model = RadioMambaNet(
                in_channels=model_config['in_channels'],
                out_channels=model_config['out_channels'],
                dims=model_config['dims'],
                depths=model_config['depths'],
                ssm_d_state=model_config['ssm_d_state'],
                ssm_d_conv=model_config['ssm_d_conv'],
                ssm_expand=model_config['ssm_expand']
            )
            
            # 检查设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"使用设备: {device}")
            
            # 输入输出尺寸测试
            print("\n输入输出尺寸测试:")
            test_sizes = [(256, 256), (512, 512), (224, 224)]
            
            model.eval()
            
            for h, w in test_sizes:
                try:
                    # 创建测试输入
                    test_input = torch.randn(1, model_config['in_channels'], h, w).to(device)
                    
                    # 将模型移动到设备
                    model = model.to(device)
                    
                    # 前向传播
                    with torch.no_grad():
                        output = model(test_input)
                    
                    print(f"  输入: {test_input.shape} -> 输出: {output.shape}")
                    
                except Exception as e:
                    print(f"  测试尺寸 ({h}, {w}) 失败: {str(e)}")
                    if "mamba_ssm" in str(e).lower() or "cuda" in str(e).lower():
                        print(f"    提示: Mamba模型需要GPU和mamba-ssm包支持")
            
            # 模型架构信息
            print(f"\n模型架构信息:")
            print(f"  编码器阶段数: {len([s for s in model.encoder_stages if isinstance(s, nn.ModuleList)])}")
            print(f"  解码器阶段数: {len(model.decoder_stages)//2}")
            print(f"  瓶颈层块数: {len(model.bottleneck)}")
            
        except Exception as e:
            if "mamba_ssm" in str(e).lower():
                print("错误: 需要安装 mamba-ssm 包")
                print("请运行: pip install mamba-ssm")
            else:
                print(f"模型测试失败: {str(e)}")
        
        # 训练配置信息
        print("\n" + "=" * 80)
        print("训练配置信息")
        print("=" * 80)
        training_config = config['training']
        print(f"学习率: {training_config['learning_rate']}")
        print(f"权重衰减: {training_config['weight_decay']}")
        print(f"损失函数: {training_config['criterion']}")
        print(f"损失权重: {training_config['loss_weights']}")
        print(f"学习率调度耐心值: {training_config['lr_scheduler_patience']}")
        print(f"早停耐心值: {training_config['early_stopping_patience']}")
        print(f"最大训练步数: {config['trainer_config']['max_steps']}")
            
    else:
        print(f"配置文件不存在: {config_path}")
        print("请检查文件路径是否正确")