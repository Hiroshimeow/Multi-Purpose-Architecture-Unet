# các t
import argparse
import json
import yaml
import torch
from pathlib import Path
from thop import profile
from src.models import get_model

def calculate_and_save_flops(run_path_str: str):
    run_path = Path(run_path_str)

    # 1. Kiểm tra các file cần thiết
    config_path = run_path / 'config.yaml'
    metrics_path = run_path / 'final_metrics.json'

    if not config_path.exists():
        print(f"Lỗi: Không tìm thấy config.yaml trong {run_path}. Bỏ qua.")
        return

    if not metrics_path.exists():
        print(f"Thông tin: Không tìm thấy final_metrics.json trong {run_path}. Bỏ qua.")
        return

    print(f"Đang xử lý lần chạy: {run_path.name}")

    # 2. Đọc config và tái tạo model
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_name = config['model']['name']
        model_params = config['model']['params']
        model = get_model(model_name, model_params)
        model.eval()

    except Exception as e:
        print(f"  - Lỗi khi tải model: {e}. Bỏ qua.")
        return

    # 3.  FLOPs
    try:
        patch_size = config.get('data', {}).get('patching', {}).get('patch_size', 224)
        in_channels = config.get('model', {}).get('params', {}).get('in_channels', 25)
        
        print(f"  - Creating dummy input with shape: (1, {in_channels}, {patch_size}, {patch_size})")
        dummy_input = torch.randn(1, in_channels, patch_size, patch_size)

        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = flops / 1e9
        print(f"  - GFLOPs : {gflops:.2f}")

    except Exception as e:
        print(f"  - Lỗi trong quá trình tính FLOPs: {e}. Bỏ qua.")
        return

    # 4. Đọc, cập nhật và ghi lại file metrics
    try:
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

        # Đảm bảo key 'performance' tồn tại
        if 'performance' not in metrics_data:
            metrics_data['performance'] = {}

        # Cập nhật GFLOPs
        metrics_data['performance']['gflops'] = gflops

        # Tính và cập nhật Efficiency
        best_miou = metrics_data.get('best_miou_from_training', 0)
        efficiency = best_miou / gflops if gflops > 0 else 0
        metrics_data['performance']['efficiency'] = efficiency
        print(f"  - Efficiency (mIoU/GFLOPs): {efficiency:.4f}")

        # Ghi lại file
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        
        print(f"  - Đã cập nhật thành công {metrics_path.name}")

    except Exception as e:
        print(f"  - Lỗi khi đọc/ghi file metrics: {e}. Bỏ qua.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tính toán FLOPs cho một model đã huấn luyện và cập nhật file metrics của nó.")
    parser.add_argument('run_path', type=str, help='Đường dẫn đến thư mục của lần chạy huấn luyện (ví dụ: training_runs/some_run_name).')
    args = parser.parse_args()

    calculate_and_save_flops(args.run_path)
