#!/usr/bin/env python3
"""测试 SS2D 导入"""
import sys
import os

# 添加路径
efficient_vmamba_path = '/home/zqm1/workspace2/EfficientVMamba-main/classification'
sys.path.insert(0, efficient_vmamba_path)
sys.path.insert(0, os.path.join(efficient_vmamba_path, 'models'))

try:
    from models.vmamba_efficient import SS2D
    print("✓ Success: SS2D imported from models.vmamba_efficient")
except ImportError as e1:
    print(f"✗ Failed to import from models.vmamba_efficient: {e1}")
    try:
        from vmamba_efficient import SS2D
        print("✓ Success: SS2D imported from vmamba_efficient")
    except ImportError as e2:
        print(f"✗ Failed to import from vmamba_efficient: {e2}")
        print("\n请确保：")
        print("1. EfficientVMamba 路径正确")
        print("2. 已安装依赖: pip install einops")
