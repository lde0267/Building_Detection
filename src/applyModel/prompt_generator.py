import numpy as np
import cv2
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from rasterio.transform import rowcol
import pandas as pd

# ✅ 3. EPSG좌표 이미지 좌표로 변환함수
def real_to_image_coordinates(image_path, real_coordinates):
    with rasterio.open(image_path) as dataset:
        transform = dataset.transform  # TIFF 변환 정보

    # 실제 좌표를 이미지의 픽셀 좌표로 변환
    point_coords = np.array([rowcol(transform, x, y) for x, y in real_coordinates])
    point_coords = np.flip(point_coords, axis=1)  # row, col → col, row 변환

    # 이미지 범위 내에서 좌표를 클리핑
    point_coords = np.clip(point_coords, (0, 0), (dataset.width - 1, dataset.height - 1))

    return point_coords

# 그림자 지역 감지
def detect_shadow_regions(image, brightness_threshold=40, saturation_threshold=30, black_threshold=40, min_area=50):
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    l_channel = image_hls[:, :, 1]  # Lightness 채널
    s_channel = image_hls[:, :, 2]  # Saturation 채널

    # 낮은 밝기와 채도 기준으로 그림자 감지
    shadow_mask = np.zeros_like(l_channel, dtype=np.uint8)
    shadow_mask[(l_channel < brightness_threshold) & (s_channel < saturation_threshold)] = 255

    # 거의 검정색 영역 감지
    black_regions = np.all(image <= black_threshold, axis=-1)
    shadow_mask[black_regions] = 255

    #  # 작은 그림자 영역 필터링
    # contours, _ = cv2.findContours(shadow_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # shadow_mask_filtered = np.zeros_like(shadow_mask)

    # for contour in contours:
    #     if cv2.contourArea(contour) > min_area:
    #         cv2.drawContours(shadow_mask_filtered, [contour], -1, 255, thickness=cv2.FILLED)

    return shadow_mask

def load_tif_and_polygon(tifPath, polyPath, digitPath):
    # TIFF 이미지 및 폴리곤 불러오기
    image = cv2.imread(tifPath)
    with rasterio.open(tifPath) as dataset:
        bounds = dataset.bounds
        crs = dataset.crs
    polygon_gdf = gpd.read_file(polyPath)
    digital_gdf = gpd.read_file(digitPath)
    
    return image, bounds, crs, polygon_gdf, digital_gdf

def create_points(bounds, space=1.5):
    # 일정 간격으로 포인트 생성
    x_coords = np.arange(bounds.left, bounds.right, space)
    y_coords = np.arange(bounds.bottom, bounds.top, space)
    points = [Point(x, y) for x in x_coords for y in y_coords]
    
    return points

def classify_points_exclude_shadow(points_gdf, polygon_union, digital_union, shadow_points, min_distance, max_distance):
    # 1. polygon_union과 digital_union을 합친 영역 속에 존재하는 포인트를 positive로 정의
    combined_polygon = polygon_union.union(digital_union)
    positive_points = list(points_gdf[points_gdf.geometry.within(combined_polygon)].geometry)  # polygon만 positive

    # 2. shadow points는 positive에서 제외 (negative로 처리하지 않음)
    positive_points = [point for point in positive_points if (point.x, point.y) not in shadow_points]

    # 3. 일정 거리 이하인 점들을 negative로 분류하고, 일정 거리 이상인 점들은 outside points로 분류
    negative_points = []
    for point in points_gdf.geometry:
        if not point.within(combined_polygon) and combined_polygon.distance(point) >= max_distance:
            continue  # 일정 거리 이상은 outside points
        elif not point.within(combined_polygon) and min_distance <= combined_polygon.distance(point) < max_distance:
            negative_points.append(point)  # min_distance 이상 max_distance 미만은 negative로 분류

    # "outside points"는 positive와 negative 둘 다 포함하지 않는 점
    outside_points = points_gdf[~points_gdf.geometry.isin(positive_points + negative_points)]
    
    return positive_points, negative_points, outside_points

def visualize_points(polygon_gdf, digital_gdf, points_gdf, positive_points, negative_points, outside_points, crs):
    # 시각화
    fig, ax = plt.subplots(figsize=(8, 6))

    # 폴리곤 시각화
    polygon_gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1, label="Polygon")
    digital_gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1, label="Digital Polygon")

    # 모든 포인트 시각화
    # points_gdf.plot(ax=ax, color='blue', markersize=1, alpha=0.5, label="All Points")

    # Positive 포인트 시각화
    positive_gdf = gpd.GeoDataFrame(geometry=positive_points, crs=crs)
    positive_gdf.plot(ax=ax, color='green', markersize=7, alpha=0.7, label="Positive Points")

    # Negative 포인트 시각화
    negative_gdf = gpd.GeoDataFrame(geometry=negative_points, crs=crs)
    negative_gdf.plot(ax=ax, color='red', markersize=7, alpha=0.7, label="Negative Points")

    # 포함되지 않는 포인트 시각화 (positive와 negative 둘 다 포함되지 않은 점)
    outside_gdf = gpd.GeoDataFrame(geometry=outside_points.geometry, crs=crs)
    outside_gdf.plot(ax=ax, color='gray', markersize=7, alpha=0.7, label="Neutral Points")

    plt.legend(loc='lower right')
    plt.title("Create and Classitfy Points")
    plt.show()

def createPoints(space=2, tifPath=None, polyPath=None, digitPath=None, min_distance=3, max_distance=5):
    # TIFF 이미지 및 폴리곤 불러오기
    image, bounds, crs, polygon_gdf, digital_gdf = load_tif_and_polygon(tifPath, polyPath, digitPath)

    # 포인트 생성
    points = create_points(bounds, space)
    points_gdf = gpd.GeoDataFrame(geometry=points, crs=crs)

    # 그림자 지역 감지 및 그림자 포인트 제외
    shadow_mask = detect_shadow_regions(image)
    shadow_points = []

    with rasterio.open(tifPath) as dataset:
        for point in points_gdf.geometry:
            x, y = point.x, point.y
            row, col = dataset.index(x, y)
            if 0 <= row < dataset.height and 0 <= col < dataset.width and shadow_mask[row, col] == 255:
                shadow_points.append((x, y))

    # 포인트 분류
    positive_points, negative_points, outside_points = classify_points_exclude_shadow(points_gdf, polygon_gdf.geometry.unary_union, digital_gdf.geometry.unary_union, shadow_points, min_distance, max_distance)

    # 시각화
    visualize_points(polygon_gdf, digital_gdf, points_gdf, positive_points, negative_points, outside_points, crs)

    return positive_points, negative_points