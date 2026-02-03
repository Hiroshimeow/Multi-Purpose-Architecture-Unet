import torch
import time
import argparse
import sys
import os
import yaml
from pathlib import Path

# Thêm đường dẫn project vào path để import được src
sys.path.append(os.getcwd())

from src.models import get_model
from src.models.deploy import DeployTABS

def benchmark_fps(model, input_shape=(1, 5, 224, 224), device='cuda', num_runs=100, num_warmup=20):
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape).to(device)
    
    print(f"   Input shape: {input_shape}")
    print(f"   Device: {device}")

    # Warmup
    print(f"   Warmup {num_warmup} times...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    if device == 'cuda': torch.cuda.synchronize()
    
    # Benchmark
    print(f"   Running {num_runs} iterations...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    if device == 'cuda': torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = num_runs / total_time
    latency = (total_time / num_runs) * 1000
    
    return fps, latency

def main():
    parser = argparse.ArgumentParser(description="Convert TABS to DeployTABS and Benchmark")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to TABS .pth checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml (optional)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print("\n=== TABS DEPLOYMENT CONVERTER & BENCHMARK ===\n")

    # 1. Cấu hình Model
    # Giả định cấu hình lightweight của bạn
    model_params = {
        'in_channels': 5,      # Input đã cắt sẵn
        'num_classes': 5,
        'initial_filters': 6,  # Lightweight
    }

    print(f"1. Initializing DeployTABS model with params: {model_params}")
    deploy_model = DeployTABS(**model_params)
    
    # 2. Load Weights (Nếu có)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"2. Loading weights from {args.checkpoint}...")
        try:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            
            # Xử lý các format checkpoint khác nhau
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Lọc và đổi tên keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('unet.'):
                    # Giữ nguyên prefix 'unet.' vì DeployTABS cũng có self.unet
                    new_state_dict[k] = v
                elif k.startswith('band_selector'):
                    # Bỏ qua weights của selector
                    pass
                else:
                    # Các trường hợp khác (nếu cấu trúc phẳng)
                    # Nếu DeployTABS có cấu trúc self.unet, ta có thể cần thêm prefix
                    new_state_dict[f'unet.{k}'] = v
            
            # Load vào model mới (strict=False để an toàn với các key lạ)
            msg = deploy_model.load_state_dict(new_state_dict, strict=False)
            print(f"   Weights loaded. Missing keys (expected): {msg.missing_keys}")
            print("   (Lưu ý: Nếu missing keys là empty hoặc không quan trọng, việc load thành công)")
            
            # Thử trích xuất selected bands nếu có trong checkpoint hoặc model cũ
            # Phần này chỉ mang tính tham khảo, vì logic chọn nằm ở BandSelector cũ
            print("   [INFO] Checkpoint loaded. In deployment, you must feed the specific 5 bands identified during training.")

        except Exception as e:
            print(f"   Error loading checkpoint: {e}")
            print("   Continuing with random initialization for Benchmark...")
    else:
        print("2. No checkpoint provided. Using Random Initialization for Benchmark.")

    # 3. Benchmark
    print("\n3. Benchmarking Inference Speed (DeployTABS)...")
    fps, latency = benchmark_fps(deploy_model, device=args.device)
    
    print("\n" + "="*40)
    print(f"RESULTS (DeployTABS - 5 Input Channels)")
    print("="*40)
    print(f"FPS     : {fps:.2f}")
    print(f"Latency : {latency:.2f} ms")
    print("="*40)
    print("\nKết luận: Đây là tốc độ thực tế khi deploy, loại bỏ overhead của Band Selection (25->5).")
    print("Tốc độ này có thể so sánh trực tiếp với Baseline (Entropy/SVM) vì cùng Input size.")

if __name__ == "__main__":
    main()
