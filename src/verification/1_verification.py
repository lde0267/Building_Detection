import geopandas as gpd
import re
import os
import glob
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import LineString
from scipy.spatial import cKDTree

# 1. 수치지도 폴리곤(참조 데이터)과 추론 폴리곤 로드
def load_shapefiles(ground_truth_path, predicted_path):
    gt_gdf = gpd.read_file(ground_truth_path)  # 수치지도 폴리곤
    pred_gdf = gpd.read_file(predicted_path)  # 추론한 폴리곤
    return gt_gdf, pred_gdf

# 2-1. Overlap 계산 함수 (GT 기준, 추론 기준)
def calculate_overlap(gt_gdf, pred_gdf):
    total_gt_area = gt_gdf.area.sum()  # 전체 수치지도 면적
    total_pred_area = pred_gdf.area.sum()  # 전체 추론 면적

    intersection_polygons = []  # 겹친 폴리곤 저장

    for gt_poly in gt_gdf.geometry:
        for pred_poly in pred_gdf.geometry:
            if gt_poly.intersects(pred_poly):
                intersection_polygons.append(gt_poly.intersection(pred_poly))
    
    total_overlap_area = unary_union([poly for poly in intersection_polygons]).area

    overlap_ratio_gt = total_overlap_area / total_gt_area if total_gt_area > 0 else 0
    overlap_ratio_pred = total_overlap_area / total_pred_area if total_pred_area > 0 else 0
    return overlap_ratio_gt, overlap_ratio_pred

# 2-2. IoU 계산 함수
def calculate_iou(gt_gdf, pred_gdf):
    # 전체 GT와 Prediction의 합집합
    union_polygon = unary_union(gt_gdf.geometry).union(unary_union(pred_gdf.geometry))
    total_union_area = union_polygon.area
    # Overlap area 계산 (기존 overlap 함수 재활용)
    _, _ = None, None  # placeholder, 직접 계산 필요
    intersection_polygons = []
    for gt_poly in gt_gdf.geometry:
        for pred_poly in pred_gdf.geometry:
            if gt_poly.intersects(pred_poly):
                intersection_polygons.append(gt_poly.intersection(pred_poly))
                
    total_overlap_area = sum([poly.area for poly in intersection_polygons])
    iou_ratio = total_overlap_area / total_union_area if total_union_area > 0 else 0
    return iou_ratio

# 2-3. BIoU (Boundary IoU) 계산 함수
def calculate_biou(gt_gdf, pred_gdf, boundary_buffer=2.0):
    gt_boundary = unary_union([poly.boundary.buffer(boundary_buffer) for poly in gt_gdf.geometry])
    pred_boundary = unary_union([poly.boundary.buffer(boundary_buffer) for poly in pred_gdf.geometry])
    boundary_intersection = gt_boundary.intersection(pred_boundary).area
    boundary_union = gt_boundary.union(pred_boundary).area
    biou_ratio = boundary_intersection / boundary_union if boundary_union > 0 else 0
    return biou_ratio

# 2-4. Precision, F1 score 계산
# Precision 및 F1-score 계산 함수
def calculate_precision_f1(gt_gdf, pred_gdf):
    # True Positive (TP): GT와 Prediction이 겹치는 영역
    intersection_polygons = [
        gt_poly.intersection(pred_poly)
        for gt_poly in gt_gdf.geometry
        for pred_poly in pred_gdf.geometry
        if gt_poly.intersects(pred_poly)
    ]
    TP = sum(poly.area for poly in intersection_polygons)

    # False Positive (FP): Prediction에서 GT와 겹치지 않는 영역
    pred_union = unary_union(pred_gdf.geometry)  # 전체 Prediction 영역
    gt_union = unary_union(gt_gdf.geometry)     # 전체 GT 영역
    
    FP = pred_union.difference(gt_union).area  # Predicted 영역에서 GT와 겹치지 않는 부분
    FN = gt_union.difference(pred_union).area  # GT 영역에서 Predicted와 겹치지 않는 부분

    # Precision 계산
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Recall 계산
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # F1-score 계산
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


# 전체 지표를 한번에 계산하는 함수
def calculate_indicators(gt_gdf, pred_gdf, pred_path, boundary_buffer=2.0):
    # Overlap 계산
    overlap_ratio_gt, overlap_ratio_pred = calculate_overlap(gt_gdf, pred_gdf)
    # IoU 계산
    iou_ratio = calculate_iou(gt_gdf, pred_gdf)
    # BIoU 계산
    biou_ratio = calculate_biou(gt_gdf, pred_gdf, boundary_buffer)

    precision, recall, f1_score = calculate_precision_f1(gt_gdf, pred_gdf)

    index_name = os.path.splitext(os.path.basename(pred_path))[0]
    
    indicators = {
        "index" : index_name,
        "overlap_ratio_gt": overlap_ratio_gt,
        "overlap_ratio_pred": overlap_ratio_pred,
        "iou_ratio": iou_ratio,
        "biou_ratio": biou_ratio,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

    return indicators

print("is it work?")

# 현재 src 폴더 기준으로 data 폴더 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # src의 부모 디렉토리 (즉, buildingDetection/)
DATA_DIR = os.path.join(BASE_DIR, "data")

# 파일 경로 설정
each_infer_folder = os.path.normpath(os.path.join(DATA_DIR, "for_paper/gangseo_underseg/undersegPoly2"))
each_digital_folder = os.path.normpath(os.path.join(DATA_DIR, "for_paper/gangseo_underseg/GT2"))
each_samPoly_folder = os.path.normpath(os.path.join(DATA_DIR, "for_paper/jungrang_samPoly"))

mask_shapefiles = glob.glob(os.path.join(each_samPoly_folder, "samPoly*.shp"))
results = []

# 5. 모든 sam추론 파일에 대해 Overlap 및 IoU 비율 계산
for mask_path in mask_shapefiles:
    mask_filename = os.path.basename(mask_path)  # 예: "polygon_0_digital.shp"

    # 숫자 부분 추출
    num_part = re.search(r'\d+', mask_filename).group()
    
    # 예상되는 추론 폴리곤과 마스크 폴리곤 파일명 생성
    infer_filename = f"underSegPoly{num_part}.shp"
    gt_filename = f"digitalPoly{num_part}.shp"

    each_pred_path = os.path.join(each_infer_folder, infer_filename)
    gt_path = os.path.join(each_digital_folder, gt_filename)

    # 1️⃣ GT vs 추론 폴리곤 비교
    if os.path.exists(each_pred_path):
        gt_gdf, each_pred_gdf = load_shapefiles(gt_path, each_pred_path)
        indicators = calculate_indicators(gt_gdf, each_pred_gdf, each_pred_path, boundary_buffer=2.0)
        results.append(indicators)
    else:
        print(f"[경고] {infer_filename} 파일을 찾을 수 없습니다. ({each_pred_path})")

    # 2️⃣ GT vs 마스크 폴리곤 비교
    if os.path.exists(mask_path):
        gt_gdf, mask_pred_gdf = load_shapefiles(gt_path, mask_path)
        
        # 하나로 합쳐진 마스크 폴리곤을 GeoDataFrame으로 변환
        unified_mask = mask_pred_gdf.unary_union
        unified_mask_gdf = gpd.GeoDataFrame(geometry=[unified_mask], crs=mask_pred_gdf.crs)

        indicators = calculate_indicators(gt_gdf, unified_mask_gdf, mask_path)
        results.append(indicators)
    else:
        print(f"[경고] {mask_filename} 파일을 찾을 수 없습니다. ({mask_path})")

# Convert results into DataFrame for easy display
df_results = pd.DataFrame(results)
pd.set_option("display.max_rows", None)  # 모든 행 출력
pd.set_option("display.max_columns", None)
# Print the results as a table
print(df_results)

#데이터프레임 분석
paired_data = []
for i in range(0, len(df_results), 2):  # 두 개씩 묶음
    underseg = df_results.iloc[i]
    sam = df_results.iloc[i + 1]

    # 인덱스 변경
    index_name = underseg["index"].replace("underSegPoly", "Poly")

    # 증가/감소 계산
    iou_change = sam["iou_ratio"] - underseg["iou_ratio"]
    precision_change = sam["precision"] - underseg["precision"]
    recall_change = sam["recall"] - underseg["recall"]
    f1_change = sam["f1_score"] - underseg["f1_score"]

    paired_data.append({
        "index": index_name,
        "iou_ratio_change": iou_change,
        "precision_change": precision_change,
        "recall_change": recall_change,
        "f1_score_change": f1_change
    })

# 결과 데이터프레임 생성
diff_df = pd.DataFrame(paired_data)
print(diff_df)

print(f"iou 증감 평균값 : {diff_df['iou_ratio_change'].mean()}")
print(f"precision 증감 평균값 : {diff_df['precision_change'].mean()}")
print(f"recall 증감 평균값 : {diff_df['recall_change'].mean()}")
print(f"f1 score 증감 평균값 : {diff_df['f1_score_change'].mean()}")

import pandas as pd

# 결과 데이터프레임을 엑셀로 저장
with pd.ExcelWriter('results_comparison_gangseo.xlsx') as writer:
    # 첫 번째 시트에 df_results 저장
    df_results.to_excel(writer, sheet_name='Results', index=False)
    
    # 두 번째 시트에 diff_df 저장
    diff_df.to_excel(writer, sheet_name='Differences', index=False)

print("엑셀 파일로 저장되었습니다.")
