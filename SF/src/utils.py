# src/utils.py
import numpy as np
from scipy.ndimage import distance_transform_edt
import cv2

def generate_boundary_weight_map(mask: np.ndarray, w0: float = 10.0, sigma: float = 5.0) -> np.ndarray:
    """
    Generates a weight map that gives higher weights to pixels near boundaries
    between different classes, as described in the U-Net paper.

    Args:
        mask (np.ndarray): The ground truth mask of shape (H, W) with integer class labels.
        w0 (float): A factor to control the magnitude of the boundary weights.
        sigma (float): A parameter to control the spread of the Gaussian penalty.

    Returns:
        np.ndarray: A weight map of shape (H, W) with higher values at boundaries.
    """
    # Chuyển mask sang kiểu dữ liệu phù hợp để xử lý
    mask = mask.astype(np.int32)
    
    # Tìm các đường biên giữa các lớp khác nhau.
    # Cách 1: Dùng Canny edge detection trên mask đã được scale
    # Cách 2: Dùng morphological operations (hiệu quả hơn cho mask)
    boundaries = np.zeros_like(mask, dtype=np.uint8)
    unique_labels = np.unique(mask)
    
    for label in unique_labels:
        if label == 0: # Bỏ qua background nếu cần
            continue
        
        class_mask = (mask == label).astype(np.uint8)
        
        # Dilation - Erosion = Boundary
        dilated = cv2.dilate(class_mask, np.ones((3, 3), np.uint8), iterations=1)
        eroded = cv2.erode(class_mask, np.ones((3, 3), np.uint8), iterations=1)
        boundary = dilated - eroded
        boundaries[boundary > 0] = 1

    # Nếu không có đường biên nào, trả về bản đồ trọng số 1
    if np.sum(boundaries) == 0:
        return np.ones_like(mask, dtype=np.float32)

    # Tính khoảng cách từ mỗi pixel đến đường biên gần nhất
    # distance_transform_edt tính khoảng cách đến background (giá trị 0)
    # nên chúng ta cần đảo ngược bản đồ đường biên
    distances = distance_transform_edt(1 - boundaries)

    # Tính trọng số theo công thức Gaussian
    # w(x) = 1 + w0 * exp(-d(x)^2 / (2 * sigma^2))
    # Thêm 1 để đảm bảo trọng số cơ bản là 1, không phải 0
    weights = 1.0 + w0 * np.exp(-(distances**2) / (2 * sigma**2))
    
    return weights.astype(np.float32)