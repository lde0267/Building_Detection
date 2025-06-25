import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

mergedSAM_path = os.path.normpath(os.path.join(DATA_DIR, "detection_result/jungrang/2022/jungrang_Building_Segformer_synthetic_deepness.shp"))
mergedDigital_path = os.path.normpath(os.path.join(DATA_DIR, "digital_map/jungrang/2022/jungrang_2022_kimyunhye_reviewed.shp"))
mergedDigital_path = os.path.normpath(os.path.join(DATA_DIR, "each_digital/jungrang/merged_digitalPoly.shp"))
output_file = os.path.normpath(os.path.join(DATA_DIR, "sam_poly/jungrang_margin60/goodresult_samPoly.shp"))

def calculate_iou(poly1, poly2):
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union if union > 0 else 0

def separate_polygons(geo_df):
    """
    GeoDataFrame 내 MultiPolygon을 개별 Polygon으로 분리
    """
    polygons = []
    for geom in geo_df.geometry:
        if isinstance(geom, MultiPolygon):
            polygons.extend(list(geom.geoms))  # MultiPolygon을 개별 Polygon으로 분리
        elif isinstance(geom, Polygon):
            polygons.append(geom)  # 이미 Polygon이면 그대로 추가
    return gpd.GeoDataFrame(geometry=polygons, crs=geo_df.crs)

def extract_high_iou_polygons(merged_samPoly_path, reference_map_path, iou_threshold=0.5):
    if not os.path.exists(merged_samPoly_path) or not os.path.exists(reference_map_path):
        print("❌ 오류: 하나 이상의 Shapefile이 존재하지 않습니다.")
        return

    # 파일 로드
    merged_samPoly = gpd.read_file(merged_samPoly_path).dropna(subset=['geometry'])
    reference_map = gpd.read_file(reference_map_path).dropna(subset=['geometry'])

    # 좌표계 변환 (필요 시)
    if merged_samPoly.crs != reference_map.crs:
        print("⚠️ 좌표계가 다릅니다. 변환을 수행합니다.")
        reference_map = reference_map.to_crs(merged_samPoly.crs)

    # 병합된 수치지도 폴리곤을 개별 폴리곤으로 분리
    merged_samPoly = separate_polygons(merged_samPoly)
    print("digital")
    reference_map = separate_polygons(reference_map)

    # 폴리곤 개수 출력
    print(f"총 추론 폴리곤 개수: {len(merged_samPoly)}")
    print(f"총 분리된 수치지도 폴리곤 개수: {len(reference_map)}")

    # IoU 기준 이상 겹치는 폴리곤 필터링
    matched_polygons = []
    for ref_poly in reference_map.geometry:
        for sam_poly in merged_samPoly.geometry:
            iou = calculate_iou(ref_poly, sam_poly)
            if iou >= iou_threshold:
                matched_polygons.append(sam_poly)
                break  # 하나만 매칭되면 멈춤

    # 매칭된 폴리곤 개수 출력
    print(f"✅ 매칭된 폴리곤 개수: {len(matched_polygons)}")

    if not matched_polygons:
        print("⚠️ 매칭된 폴리곤이 없습니다. 파일을 저장하지 않습니다.")
        return

    # 결과 저장
    matched_gdf = gpd.GeoDataFrame(geometry=matched_polygons, crs=merged_samPoly.crs)
    matched_gdf.to_file(output_file, driver='ESRI Shapefile', encoding='utf-8')

# 사용 예시
extract_high_iou_polygons(mergedSAM_path, mergedDigital_path, iou_threshold=0.8 )
