import cv2
import os
import numpy as np

def crop_by_result(result, filename, output_dir='output_crops', label_map=None):
    """
    YOLO/SAM 결과를 기반으로 이미지를 Crop하여 저장하고,
    해당 객체들의 Raw Mask 데이터를 인덱스 기반 딕셔너리로 반환하는 함수.
    
    Args:
        result: Ultralytics 모델 예측 결과 객체
        filename: 원본 이미지 파일명
        output_dir: 저장할 루트 디렉토리
        label_map: (사용 안 함, 호환성 유지용)
        
    Returns:
        mask_dict: { 객체ID(int): mask_boolean_array } 
                   -> 추후 track_crop에서 실제 클래스명으로 매핑됨
    """
    
    orig_img = result.orig_img
    h, w, _ = orig_img.shape

    # 반환할 딕셔너리 초기화
    mask_dict = {}

    # 탐지된 객체가 없으면 빈 딕셔너리 반환
    if result.masks is None:
        return mask_dict

    # 데이터 추출
    # masks: (N, H, W) - 모델 출력 해상도의 마스크
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy()

    # 중복 저장 방지용 (한 프레임 내에서 완벽히 동일한 크기의 객체 방지)
    seen = []

    for i, mask in enumerate(masks):
        # ---------------------------------------------------------
        # [Step 1] 저장될 폴더명 결정 (여기선 임시로 ID 사용)
        # ---------------------------------------------------------
        # track_crop.py에서 obj_ids를 [0, 1, 2...] 순서로 넣었으므로
        # 여기서 i(인덱스)가 곧 객체의 추적 ID가 됩니다.
        class_name = str(i)

        # ---------------------------------------------------------
        # [Step 2] 마스크 데이터 저장 (3D 하이라이트용 핵심 로직)
        # ---------------------------------------------------------
        # 마스크 리사이징 (모델 출력이 원본과 다를 경우 대비)
        if mask.shape != (h, w):
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask
            
        # Boolean Mask 저장 (메모리 절약을 위해 bool 타입 권장)
        mask_bool = mask_resized > 0.5
        mask_dict[i] = mask_bool

        # ---------------------------------------------------------
        # [Step 3] 이미지 파일 저장 (IR 및 갤러리용)
        # ---------------------------------------------------------
        # 시각화를 위해 uint8 변환
        mask_u8 = mask.astype(np.uint8)
        resized_mask = cv2.resize(mask_u8, (w, h))
        binary_mask = (resized_mask > 0.5).astype(np.uint8) * 255

        # 투명 배경 이미지 생성 (RGBA)
        b, g, r = cv2.split(orig_img)
        rgba_img = cv2.merge([b, g, r, binary_mask])

        # Bounding Box 좌표 추출 및 클리핑
        if i < len(boxes):
            x1, y1, x2, y2 = boxes[i].astype(int)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            
            # 이미지 Crop
            crop_img = rgba_img[y1:y2, x1:x2]

            # 파일 저장
            if crop_img.shape not in seen:
                seen.append(crop_img.shape)
                
                # 1. 폴더 생성 (폴더명 = "0", "1"...)
                class_dir = os.path.join(output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                base_name = os.path.splitext(os.path.basename(filename))[0]
                file_name = f"{base_name}_{i}.png"
                save_path = os.path.join(class_dir, file_name)
                
                # 이미지 저장
                cv2.imwrite(save_path, crop_img)

    return mask_dict