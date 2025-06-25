import os
import pandas as pd
import shutil

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # src의 부모 디렉토리 (즉, buildingDetection/)
DATA_DIR = os.path.join(BASE_DIR, "data")

excel_path = os.path.normpath(os.path.join(BASE_DIR, "merged_output.xlsx"))
each_samPoly_folder = os.path.normpath(os.path.join(DATA_DIR, "sam_poly/jungrang_margin60"))
print(each_samPoly_folder)
destination_folder = os.path.normpath(os.path.join(DATA_DIR, "sam_poly/select_underSeg")) 

# 복사할 폴더가 없다면 생성
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

sheet1 = pd.read_excel(excel_path, sheet_name=0)

# 'index'가 'UnderSeg'로 시작하는 행 필터링 및 'iou_ratio'로 정렬
filtered_df = sheet1[sheet1["index_sheet1"].str.startswith("underSegPoly")].sort_values(by="iou_ratio", ascending=True)

# 0.7이하일 경우 모두 추출
# filtered_df = filtered_df[(filtered_df["iou_ratio"] <= 0.85) & (filtered_df["iou_ratio"] >= 0.3)]
filtered_df = filtered_df[(filtered_df["iou_ratio"] >= 0.3)]


# 결과 출력
print(f"총 {len(filtered_df)}개의 행이 조건을 만족합니다.")
print(filtered_df.head(20))

# 각 행의 'index_sheet1' 값에 따라 파일 복사
for _, row in filtered_df.iterrows():
    # 'underSegPoly' 부분을 제거하고, 나머지 파일 이름을 추출한 후 확장자 추가
    base_file_name = row['index_sheet1'].replace("underSegPoly", "samPoly")
    
    # 복사할 파일들 (.shp, .shx, .dbf, .prj, .cpg)
    extensions = [".shp", ".shx", ".dbf", ".prj", ".cpg"]
    
    for ext in extensions:
        file_name = base_file_name + ext
        source_file = os.path.join(each_samPoly_folder, file_name)  # 원본 파일 경로
        destination_file = os.path.join(destination_folder, file_name)  # 목적지 파일 경로

        # 파일이 존재하는지 확인 후 복사
        if os.path.exists(source_file):
            shutil.copy(source_file, destination_file)
            print(f"파일 '{file_name}'이(가) '{destination_folder}'로 복사되었습니다.")
        else:
            print(f"파일 '{file_name}'을 찾을 수 없습니다.")