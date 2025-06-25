import geopandas as gpd
import os
import glob
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.spatial import cKDTree

# 1. 수치지도 폴리곤(참조 데이터)과 추론 폴리곤 로드
def load_shapefiles(ground_truth_path, predicted_path):
    gt_gdf = gpd.read_file(ground_truth_path)  # 수치지도 폴리곤
    pred_gdf = gpd.read_file(predicted_path)  # 추론한 폴리곤
    return gt_gdf, pred_gdf

# 2. 폴리곤 매칭 (IoU 기준으로 대응되는 폴리곤을 찾기)
def match_polygons_by_iou(gt_gdf, pred_gdf):
    matched_pairs = []
    
    for idx_gt, gt_poly in gt_gdf.iterrows():
        for idx_pred, pred_poly in pred_gdf.iterrows():
            intersection = gt_poly.geometry.intersection(pred_poly.geometry)
            iou = intersection.area / (gt_poly.geometry.area + pred_poly.geometry.area - intersection.area)
            if iou > 0:  # IOU가 0보다 큰 경우만 대응
                matched_pairs.append((gt_poly, pred_poly, iou))
    
    return matched_pairs

# 3. Precision, Recall, F1, IoU 지표 계산
def calculate_metrics_for_pair(gt_poly, pred_poly):
    intersection = gt_poly.geometry.intersection(pred_poly.geometry)
    TP = intersection.area
    FP = pred_poly.geometry.area - TP
    FN = gt_poly.geometry.area - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = TP / (gt_poly.geometry.area + pred_poly.geometry.area - TP) if (gt_poly.geometry.area + pred_poly.geometry.area - TP) > 0 else 0
    
    return precision, recall, f1_score, iou

# 4. 각 폴리곤에 대해 계산한 결과 저장
def calculate_indicators_for_all_pairs(gt_gdf, pred_gdf):
    matched_pairs = match_polygons_by_iou(gt_gdf, pred_gdf)
    
    results = []
    for gt_poly, pred_poly, iou in matched_pairs:
        precision, recall, f1_score, iou = calculate_metrics_for_pair(gt_poly, pred_poly)
        
        results.append({
            "gt_index": gt_poly.name,
            "pred_index": pred_poly.name,
            "iou_ratio": iou,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        })
    
    return results

# 5. 결과를 DataFrame으로 변환하여 출력
def main():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    each_infer_folder = os.path.normpath(os.path.join(DATA_DIR, "for_paper/jungrang_underseg"))
    each_digital_folder = os.path.normpath(os.path.join(DATA_DIR, "for_paper/jungrang_digital"))
    each_samPoly_folder = os.path.normpath(os.path.join(DATA_DIR, "for_paper/jungrang_samPoly"))

    mask_shapefiles = glob.glob(os.path.join(each_samPoly_folder, "samPoly.shp"))
    results = []

    # 모든 샘플에 대해 계산
    for mask_path in mask_shapefiles:
        infer_filename = f"underSegPoly486.shp"
        gt_filename = f"digitalPoly1_2022.shp"

        each_pred_path = os.path.join(each_infer_folder, infer_filename)
        gt_path = os.path.join(each_digital_folder, gt_filename)

        if os.path.exists(each_pred_path):
            gt_gdf, each_pred_gdf = load_shapefiles(gt_path, each_pred_path)
            indicators = calculate_indicators_for_all_pairs(gt_gdf, each_pred_gdf)
            results.extend(indicators)
        else:
            print(f"[경고] {infer_filename} 파일을 찾을 수 없습니다. ({each_pred_path})")

        if os.path.exists(mask_path):
            gt_gdf, mask_pred_gdf = load_shapefiles(gt_path, mask_path)
            indicators = calculate_indicators_for_all_pairs(gt_gdf, mask_pred_gdf)
            results.extend(indicators)
        else:
            print(f"[경고] {mask_path} 파일을 찾을 수 없습니다. ({mask_path})")

    # 결과 출력
    df_results = pd.DataFrame(results)
    pd.set_option("display.max_rows", None)  # 모든 행 출력
    pd.set_option("display.max_columns", None)
    print(df_results)

    # 엑셀로 저장
    with pd.ExcelWriter('results_jungrang_2022_each.xlsx') as writer:
        df_results.to_excel(writer, sheet_name='Results', index=False)

    print("엑셀 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()
