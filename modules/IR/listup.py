import sqlite3
import unicodedata
import requests
from io import BytesIO
from PIL import Image

# 수정된 crop 함수 (track_crop.py) 임포트
from .track_crop import crop
from .IR import IR

def listup(img_path):
    """
    이미지를 crop하고, IR(유사도 검색)을 수행한 뒤,
    DB에서 쇼핑몰 URL을 찾아 반환하는 함수.
    
    Returns:
        urldict: { "Sofa_0": [url, model_name], ... }
        all_frames_masks: [ { "Sofa_0": mask_bool_array, ... }, ... ] (3D 하이라이트용)
        ordered_ids: ["Sofa_0", "Table_1", ...] (갤러리 순서 매칭용)
    """
    
    urldict = {}
    
    # ---------------------------------------------------------
    # [Step 1] Crop 실행 & 마스크 데이터 획득
    # ---------------------------------------------------------
    # track_crop.py의 crop 함수가 이제 all_frames_masks를 반환합니다.
    # 구조: 프레임별 리스트 [ { "Sofa_0": mask, "Table_1": mask }, ... ]
    all_frames_masks = crop(img_path)

    # ---------------------------------------------------------
    # [Step 2] IR 실행 (이미지 검색)
    # ---------------------------------------------------------
    # output_crops 폴더에 저장된 이미지를 바탕으로 유사도 검색 수행
    # output_result 구조예시: [{'folder_id': 'Sofa_0', 'predicted_name': 'LANDSKRONA', ...}, ...]
    output_result = IR()

    # ---------------------------------------------------------
    # [Step 3] DB 조회 및 데이터 정리
    # ---------------------------------------------------------
    conn = sqlite3.connect('modules/IR/DB/ikea_image_data_multi_category2_deleted.db')
    cursor = conn.cursor()

    # 갤러리 순서 동기화를 위한 ID 리스트
    # Gradio Gallery의 select 이벤트는 index(0, 1, 2...)를 반환하므로,
    # 이 순서와 정확히 일치하는 Object ID 리스트가 필요합니다.
    ordered_ids = []

    for i in range(len(output_result)):
        folder_id = output_result[i]['folder_id']      # 예: "Sofa_0" (track_crop에서 생성한 폴더명)
        raw_name = str(output_result[i]['predicted_name']) # 예: "LANDSKRONA"

        # 문자열 정규화 (한글 자소 분리 해결)
        target_name = raw_name.strip()
        target_name = unicodedata.normalize('NFC', target_name)

        # DB 쿼리 실행
        cursor.execute("SELECT image_url FROM products_images WHERE filename = ?", (target_name,))
        result = cursor.fetchone()

        if result:
            url = result[0]
            # print(f"Found: {url}") 
            
            # urldict에 [URL, 모델명] 저장
            urldict[folder_id] = [url, raw_name]
            
            # [중요] 갤러리에 추가될 순서대로 ID를 기록
            ordered_ids.append(folder_id)

        else:
            print(f"No result found in DB for {raw_name} (ID: {folder_id})")
       
    # ---------------------------------------------------------
    # [Step 4] 결과 반환 (3개)
    # ---------------------------------------------------------
    return urldict, all_frames_masks, ordered_ids