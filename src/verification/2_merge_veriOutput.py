import pandas as pd
import re
import os

# 엑셀 파일 읽기
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # src의 부모 디렉토리 (즉, buildingDetection/)
excel_path = os.path.normpath(os.path.join(BASE_DIR, "results_comparison_gangseo.xlsx"))


sheet1 = pd.read_excel(excel_path, sheet_name=0)
sheet2 = pd.read_excel(excel_path, sheet_name=1)

# 숫자 부분 추출 함수
def extract_number(index):
    match = re.search(r'\d+', str(index))
    return int(match.group()) if match else None

# 각 시트의 index에서 숫자 추출
df1 = sheet1.copy()
df2 = sheet2.copy()
df1["index_num"] = df1["index"].apply(extract_number)
df2["index_num"] = df2["index"].apply(extract_number)

# 같은 숫자를 가진 모든 index를 유지하여 병합
merged_df = pd.merge(df1, df2, on="index_num", how="inner", suffixes=("_sheet1", "_sheet2"))

# 결과 저장
output_path = "merged_output_gangseo.xlsx"
merged_df.to_excel(output_path, index=False)

print(f"병합된 파일이 {output_path}에 저장되었습니다.")
