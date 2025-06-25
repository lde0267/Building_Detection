기존 건물 추론 결과를 바탕으로 fastsam을 이용하여 결과를 보정하는 방법론에 관한 코드와 설명입니다.
특히 여러 건물을 하나의 건물로 추론하는 underSegmentation 문제를 해결하고자 합니다.

1. 실행하기 전 필요한 데이터는 다음과 같습니다.
   - 해당 지역 정사영상
   - 해당 지역 추론 결과(선행 연구에서 사용한 segFormer를 사용했습니다.)
   - 해당 지역 수치지도

2. 사용한 모델을 fastsam으로 FastSAM-x.pt를 다운받아 사용하였습니다.

![figs3](https://github.com/user-attachments/assets/e91321f0-ba5b-4ba6-9e22-b4e17ae1ae0b)


Descriptions or papers on the methodology can be found below.
[FASTSAM-BASED BOUNDARY REFINEMENT METHOD_LEE Dongun.pdf](https://github.com/user-attachments/files/20897303/FASTSAM-BASED.BOUNDARY.REFINEMENT.METHOD_LEE.Dongun.pdf)
