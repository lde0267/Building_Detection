import geopandas as gpd
import os

# 현재 src 폴더 기준으로 data 폴더 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # src의 부모 디렉토리 (즉, buildingDetection/)
DATA_DIR = os.path.join(BASE_DIR, "data")

# 파일 경로 설정
polygon_file = os.path.normpath(os.path.join(DATA_DIR, "All infer Polygon Folder/poly.shp")) # 기존 추론 폴리곤
digital_file = os.path.normpath(os.path.join(DATA_DIR, "All digitalMap Folder/digital.shp"))
output_underSegPoly_folder = os.path.normpath(os.path.join(DATA_DIR, "Output each underseg Polygon Folder"))
output_digitPoly_folder = os.path.normpath(os.path.join(DATA_DIR, "Output each digital Polygon Folder"))

# 저장할 폴더 생성
os.makedirs(output_underSegPoly_folder, exist_ok=True)
os.makedirs(output_digitPoly_folder, exist_ok=True)

# 데이터 로드
polygon_gdf = gpd.read_file(polygon_file)  # 모든 추론된 폴리곤이 포함된 파일
digital_map = gpd.read_file(digital_file)  # 수치지도 데이터

# 좌표계 변환 (필요하면)
if polygon_gdf.crs != digital_map.crs:
    print("좌표계가 달라")
    digital_map = digital_map.to_crs(polygon_gdf.crs)

# 개별 폴리곤 처리
count = 0  # 파일명 번호 관리
for idx, pred_poly in polygon_gdf.iterrows():
    # 현재 추론된 폴리곤과 겹치는 수치지도 폴리곤 찾기
    overlapping = digital_map[digital_map.intersects(pred_poly.geometry)]

    # 2개 이상 겹칠 경우만 저장
    if len(overlapping) >= 2:
        underSegPoly_save_path = os.path.join(output_underSegPoly_folder, f"underSegPoly{count}.shp")
        gpd.GeoDataFrame([pred_poly], crs=polygon_gdf.crs).to_file(underSegPoly_save_path)
        print(f"✅ 저장 완료: {underSegPoly_save_path}")

        merged_digi_poly = overlapping.unary_union  # 겹치는 수치지도 폴리곤들을 병합
        merged_digi_gdf = gpd.GeoDataFrame([{"geometry": merged_digi_poly}], crs=digital_map.crs)
        
        # 병합된 수치지도 폴리곤 저장
        digitPoly_save_path = os.path.join(output_digitPoly_folder, f"digitalPoly{count}.shp")
        merged_digi_gdf.to_file(digitPoly_save_path)
        print(f"✅ 저장 완료: {digitPoly_save_path}")

        count += 1  # 파일명 증가


# 저장된 폴리곤 개수 출력
if count > 0:
    print(f"✅ 총 {count}개의 UnderSeg, digital 폴리곤이 저장되었습니다.")
else:
    print("⚠️ 2개 이상 겹치는 폴리곤이 없습니다.")
