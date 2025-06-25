import os
import numpy as np
import rasterio
from ultralytics.models.fastsam import FastSAMPredictor
import rasterio
from rasterio.transform import rowcol
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# ✅ 1. FastSAM 모델 불러오기
def create_fastsam_predictor(model_filename="FastSAM-x.pt", conf=0.2, iou=0.8):
    model_path = os.path.join(os.path.dirname(__file__), model_filename)
    overrides = {
        "conf": conf,
        "iou": iou,
        "task": "segment",
        "mode": "predict",
        "model": model_path,
        "save": False
    }
    return FastSAMPredictor(overrides=overrides)

# ✅ 3. EPSG좌표 이미지 좌표로 변환함수
def real_to_image_coordinates(image_path, real_coordinates):
    with rasterio.open(image_path) as dataset:
        transform = dataset.transform  # TIFF 변환 정보

    # 실제 좌표를 이미지의 픽셀 좌표로 변환
    point_coords = np.array([rowcol(transform, point.x, point.y) for point in real_coordinates])
    point_coords = np.flip(point_coords, axis=1)  # row, col → col, row 변환

    # 이미지 범위 내에서 좌표를 클리핑
    point_coords = np.clip(point_coords, (0, 0), (dataset.width - 1, dataset.height - 1))

    return point_coords

def combine_masks(mask_list):
    # Ensure there is at least one mask
    if len(mask_list) == 0:
        return None

    # Initialize a mask with zeros of the same shape as the first mask in the list
    combined_mask = np.zeros_like(mask_list[0])

    # Apply logical OR to combine all masks
    for mask in mask_list:
        combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)

    return combined_mask

# ✅ 4. FastSAM을 사용하여 마스크 생성
def generate_fastsam_mask(image_path, positive_coords, negative_coords, samPredictor):
    # EPSG 좌표를 이미지의 픽셀 좌표로 변환
    pos_point_coords = real_to_image_coordinates(image_path, positive_coords)
    neg_point_coords = []
    if negative_coords:
        neg_point_coords = real_to_image_coordinates(image_path, negative_coords)

    predictor = samPredictor
    everything_results = predictor(image_path)

    print("🚀 Everything Results:", everything_results)

    mask_list = []

    for pos in pos_point_coords:
         # Calculate distances between the current positive point and all other positive points
        distances = euclidean_distances([pos], pos_point_coords)[0]
        
        # Get the indices of the 5 nearest points (excluding the point itself)
        nearest_indices = np.argsort(distances)[1:3]
        
        # Get the coordinates of the 5 nearest points
        nearest_pos_points = [pos_point_coords[i] for i in nearest_indices]

        # Prepare the input for the prompt (positive points from nearest_pos_points and negative points if available)
        input_points = np.array(nearest_pos_points + (list(neg_point_coords) if neg_point_coords is not None else []))
        input_labels = np.array([1] * len(nearest_pos_points) + [0] * len(neg_point_coords))

        mask_results = predictor.prompt(everything_results, points=input_points.tolist(), labels=input_labels.tolist())

        # 마스크가 존재하는지 확인 후 처리
        if mask_results and hasattr(mask_results[0], "masks") and mask_results[0].masks:
            mask_array = mask_results[0].masks.data.cpu().numpy()

            # 차원 확인 후 변환
            if mask_array.ndim == 3:
                mask_array = mask_array[0]  # 첫 번째 마스크 선택
            
            mask_list.append(mask_array)
        else:
            print("No mask detected")

    return combine_masks(mask_list)

