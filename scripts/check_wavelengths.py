import scipy.io as io
import numpy as np
import glob
import os

# Tìm file .mat đầu tiên
data_dir = "/data1/tabs/Image_dataset/cubes_fl32/"
files = glob.glob(os.path.join(data_dir, "*.mat"))

if not files:
    print("Không tìm thấy file .mat nào!")
else:
    file_path = files[0]
    print(f"Đang kiểm tra file: {file_path}")
    
    try:
        mat_data = io.loadmat(file_path)
        print("\nCác keys trong file .mat:")
        for key in mat_data.keys():
            if not key.startswith('__'):
                print(f" - {key}: {type(mat_data[key])}")
                if isinstance(mat_data[key], np.ndarray):
                    print(f"   Shape: {mat_data[key].shape}")
                
        # Kiểm tra wavelength
        potential_keys = ['wavelength', 'wavelengths', 'wl', 'bands', 'band_centers']
        found = False
        for k in potential_keys:
            if k in mat_data:
                print(f"\n✅ TÌM THẤY '{k}':")
                print(mat_data[k])
                found = True
        
        if not found:
            print("\n❌ KHÔNG tìm thấy thông tin wavelength trong file .mat.")
            
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
