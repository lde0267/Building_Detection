import cv2
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import rasterio
from prompt_generator import createPoints
from apply_sam import generate_fastsam_mask

# TIFF 영상에서 변환 정보 가져오기
def get_tiff_transform(tiff_path):
    with rasterio.open(tiff_path) as dataset:
        print("TIFF 크기:", dataset.shape)
        return dataset.transform, dataset.crs

# 이미지 로드 및 이진화
def load_binary_mask(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary_mask

# 마스크 사이즈를 원본 TIF 크기에 맞게 조정하는 함수
def resize_mask_to_tif(mask, tif_path):

    if mask is None or mask.size == 0:
        print("Error : Empty mask array provided")
        return None
    # 원본 TIF 파일 크기 가져오기
    with rasterio.open(tif_path) as src:
        orig_width, orig_height = src.width, src.height

    # 마스크 원본 크기 가져오기
    # sam_height, sam_width = mask.shape[:2]
    # print(f"Original TIF size: {orig_width}x{orig_height}, SAM output size: {sam_width}x{sam_height}")

    # 마스크 크기 조정 (원본 TIF 크기에 맞게)
    resized_mask = cv2.resize(mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
    
    return resized_mask

# 마스크에서 폴리곤 추출
def mask_to_polygons(mask, transform):
    if mask is None or mask.size == 0:
        print("Error : Empty mask array provided")
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 4:  # 최소 4개의 점이 필요
            geo_coords = [rasterio.transform.xy(transform, y, x) for x, y in contour[:, 0, :]]
            polygon = Polygon(geo_coords)
            polygons.append(polygon)
    return polygons

# 폴리곤을 시각화
def visualize_polygons(mask, polygons):
    visualization = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(visualization, contours, -1, (0, 255, 0), 2)
    plt.figure(figsize=(8, 6))
    plt.imshow(visualization, cmap='gray')
    plt.title("Extracted Contours")
    plt.axis("off")
    plt.show()

# 폴리곤을 Shapefile로 저장
def save_polygons_as_shapefile(polygons, crs, output_path):
    if not polygons:
        print("Warning: No polygons to save. Shapefile not created.")
        return
    gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)
    gdf.to_file(output_path)
    print(f"폴리곤 {len(polygons)}개를 검출하여 {output_path}에 저장했습니다.")

# 전체 실행 코드
def extract_polygons_from_sam(tiff_path, poly_path, digit_path, output_file, samPredictor):
    
    # 0. 프롬프트 생성
    positive_points, negative_points = createPoints(tifPath=tiff_path, polyPath=poly_path, digitPath=digit_path)

    # 1. SAM 마스크 생성
    mask = generate_fastsam_mask(tiff_path, positive_points, negative_points, samPredictor)

    # 2. 마스크 사이즈 조정
    resized_mask = resize_mask_to_tif(mask, tiff_path)

    plt.figure(figsize=(10, 6))
    plt.imshow(resized_mask, cmap="gray")
    plt.title(f"Building Mask (FastSAM)")
    plt.axis("off")  # 축 제거
    plt.show()

    # 3. TIFF의 변환 정보 가져오기
    transform, crs = get_tiff_transform(tiff_path)

    # 4. 마스크를 TIFF 좌표계의 폴리곤으로 변환
    polygons = mask_to_polygons(resized_mask, transform)

    # 5. 시각화
    # visualize_polygons(resized_mask, polygons)

    # 6. Shapefile 저장
    save_polygons_as_shapefile(polygons, crs, output_file)