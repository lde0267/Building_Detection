import os
import geopandas as gpd
import pandas as pd
import glob

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # src의 부모 디렉토리 (즉, buildingDetection/)
DATA_DIR = os.path.join(BASE_DIR, "data")

# 파일 경로 설정
each_Poly_folder = os.path.normpath(os.path.join(DATA_DIR, "for_paper/sam_poly/sampoly2"))

# samPoly* 패턴의 모든 Shapefile 찾기
shapefiles = glob.glob(os.path.join(each_Poly_folder, "samPoly*.shp"))

# Shapefile 병합
gdfs = []
for shp in shapefiles:
    # GeoDataFrame 읽기
    gdf = gpd.read_file(shp)
    # 파일 이름을 새로운 컬럼 'source_file'에 추가
    gdf['source_file'] = os.path.basename(shp)  # 파일 이름만 추출하여 추가
    gdfs.append(gdf)

# 병합
merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))  # 병합

# 결과 저장 (출력 파일 경로 설정)
output_shapefile = os.path.join(each_Poly_folder, "merged_samPoly.shp")
merged_gdf.to_file(output_shapefile)

print(f"병합된 파일 저장 완료: {output_shapefile}")
