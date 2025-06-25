import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds
import os

# 폴리곤 파일들이 저장된 폴더 경로
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # src의 부모 디렉토리 (즉, buildingDetection/)
DATA_DIR = os.path.join(BASE_DIR, "data")

polygon_folder = os.path.normpath(os.path.join(DATA_DIR, "Each infer Polygon Folder"))
digitalMap_folder = os.path.normpath(os.path.join(DATA_DIR, "Each digitalMap Folder"))
orthophoto_path = os.path.normpath(os.path.join(DATA_DIR, "Some_Region_Drone_Image.tif"))
output_folder = os.path.normpath(os.path.join(DATA_DIR, "Each OrthoPhoto Output_Folder_Path"))

# 폴더 내 모든 Shapefile 파일을 처리
for polygon_file in os.listdir(polygon_folder):
    if polygon_file.endswith(".shp"):
        polygon_file_path = os.path.join(polygon_folder, polygon_file)
        
        # 폴리곤 파일 불러오기
        polygons = gpd.read_file(polygon_file_path)

        # 대응되는 디지털 폴리곤 파일 불러오기
        digital_file_path = os.path.join(digitalMap_folder, f"digitalPoly{polygon_file.replace('underSegPoly', '').replace('.shp', '')}.shp")
        
        if os.path.exists(digital_file_path):
            digital_polygons = gpd.read_file(digital_file_path)
            # 두 GeoDataFrame 결합 (합집합 영역 계산)
            polygons = gpd.GeoDataFrame(pd.concat([polygons, digital_polygons], ignore_index=True))

        with rasterio.open(orthophoto_path) as src:
            crs = src.crs  # 좌표계 확인
            print(f"📌 좌표계 정보: {src.crs}")
            transform = src.transform  # 변환 행렬
            
            if polygons.crs != crs:
                print(f"📌 좌표계 정보: {polygons.crs}")
                print("좌표계 정보가 다릅니다.")
            
            # === 1. 두 폴리곤의 합집합 영역 계산 ===
            union_geometry = polygons.geometry.unary_union  # 합집합 영역

            # === 2. 합집합 영역의 최소 BBox 계산 ===
            minx, miny, maxx, maxy = union_geometry.bounds  # 합집합 영역의 BBox

            # === 3. 여유 공간(Margin) 추가 ===
            margin_ratio = 0.6  # 전체 크기의 40% 여유 공간 추가
            margin_x = (maxx - minx) * margin_ratio
            margin_y = (maxy - miny) * margin_ratio

            minx -= margin_x
            maxx += margin_x
            miny -= margin_y
            maxy += margin_y

            # === 4. 정사영상에서 BBox 영역만 추출 ===
            window = from_bounds(minx, miny, maxx, maxy, transform)
            cropped_image = src.read(window=window)

            # === 5. 변환 행렬 업데이트 (BBox 기반) ===
            cropped_transform = src.window_transform(window)

        # === 6. 새로운 TIFF 파일로 저장 ===
        output_filename = polygon_file.replace("underSegPoly", "underSegOrtho").replace(".shp", ".tif")
        output_path = os.path.join(output_folder, output_filename)

        os.makedirs(output_folder, exist_ok=True)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=cropped_image.shape[1],
            width=cropped_image.shape[2],
            count=cropped_image.shape[0],  # 채널 수
            dtype=cropped_image.dtype,
            crs=src.crs,
            transform=cropped_transform,
        ) as dst:
            dst.write(cropped_image)

        print(f"✅ '{polygon_file}'와 대응되는 '{digital_file_path}'을(를) 병합하여 생성된 정사영상이 '{output_path}'에 저장되었습니다.")
