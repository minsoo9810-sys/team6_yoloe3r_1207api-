import os
from google import genai
from google.genai import types
from PIL import Image # 이미지 저장 및 처리 
import io # 바이트 스트림 처리 
from .config import API_KEY, STYLE_MODEL

def make_one_image_to_three(api_key: str, model_name: str, input_image_path: str):
    """
    앞선 과정에서 생성된 방 이미지를 입력받아,
    왼쪽 측면 뷰(Left View)와 오른쪽 측면 뷰(Right View) 2장의 이미지를 추가로 생성합니다.

    Args:
        api_key (str): Google GenAI API Key
        model_name (str): 사용할 모델명 (config.py의 STYLE_MODEL, 예: 'gemini-2.5-flash-image')
        input_image_path (str): 앞선 과정에서 생성된 원본 이미지 경로
    """
    # 1. 클라이언트 초기화.
    client = genai.Client(api_key=api_key)

    # 2. 원본 이미지 파일 읽기. (바이트 변환)
    # LLM에게 원본 이미지(레퍼런스)를 '입력'으로 제공하여, 동일한 구조와 스타일을 유지하라는 컨텍스트를 부여하기 위함
    # 즉, 1번 과정에서 만든 이미지를 모델에게 "이 공간을 기반으로 그려줘"라고 전달하기 위함.
    try:
        with open(input_image_path, "rb") as f:
            img_bytes = f.read()
    except FileNotFoundError:
        print(f" 오류: 입력 이미지를 찾을 수 없습니다. 경로를 확인하세요: {input_image_path}")
        return

    # 3. 생성할 이미지 설정. (방향, 파일명, 각도별 추가 프롬프트)

    # (1) 절대 규칙. (공통)
    base_rules = """
    당신은 실사 사진을 감쪽같이 편집하는 최고 수준의 전문 디지털 아티스트이자 포토그래머입니다.
    입력된 “단일 방 사진”을 바탕으로, 동일한 방을 유지한 상태에서 
    수평(Yaw) 각도만 다르게 적용한 실사 이미지를 생성해야 합니다.

    ────────────────────────────────────────
    절대 규칙 (어떠한 경우에도 위반 금지):
    • 출력 결과는 반드시 "단일 이미지 1장".
    • 원본 방의 구조, 물체 배치, 가구 개수, 위치, 조명 방향, 재질, 텍스처는 모두 유지.
    • 카메라의 물리적 위치 고정.
    • 수평(Yaw) 회전만 적용.
    • 상하 각도(Pitch) 변화 없음 → tilt = 0
    • 줌인/줌아웃 금지 → zoom = 0 (원본 화각 그대로 유지)
    • 출력되는 이미지는 "실사 사진(realistic photo)"이어야 하며, 3D 렌더링·그림 느낌 금지.
    • 출력 이미지의 **가로세로 비율(aspect ratio)은 반드시 입력 이미지와 동일해야 한다.**
    • Temperature = 0.1 수준의 일관성 유지.
    """

    # (2) 방향별 세부 지시. (Left / Right)
    prompts_by_direction = {
        "left": """
        [작업: 왼쪽(Yaw -30°)]
        • 입력된 사진의 방을 동일하게 유지.
        • 카메라가 제자리에서 수평으로 왼쪽으로 약 -30° 회전한 시야.
        • 사용자가 몸을 왼쪽으로 30° 돌린 시점과 동일.
        • tilt = 0, zoom = 0, 단일 프레임.

        prompt:
        camera rotated 30 degrees to the left, same room, same furniture layout,
        single frame, single viewpoint, realistic indoor photograph,
        tilt 0, zoom 0, same aspect ratio as the original image
        """,

        "right": """
        [작업: 오른쪽(Yaw +30°)]
        • 입력된 사진의 방을 동일하게 유지.
        • 카메라가 제자리에서 수평으로 오른쪽으로 약 +30° 회전한 시점.
        • 사용자가 몸만 오른쪽으로 30° 돌려 같은 방을 바라본 시야.
        • tilt = 0, zoom = 0, 단일 프레임.

        prompt:
        camera rotated 30 degrees to the right, same room, same furniture layout,
        single frame, single viewpoint, realistic indoor photograph,
        tilt 0, zoom 0, same aspect ratio as the original image
        """
    }

    # (3) 네거티브 프롬프트. (공통)
    # 생성 품질을 저해하거나 일관성을 깨는 요소를 적극적으로 배제하기 위한 지침
    negative_prompt = """
    negative prompt:
    collage, split screen, multi-view, panorama, fisheye, lens distortion,
    illustration, drawing, CGI, painting, text, watermark, subtitles,
    distorted furniture, changing room structure
    """

    # 각도 저장 파일명 : img4new3r_left.png, img4new3r_right.png.
    tasks = [
        {"direction": "left", "filename": "img4new3r_left.png"},
        {"direction": "right", "filename": "img4new3r_right.png"}
    ]

    # 4. 왼쪽/오른쪽 이미지 생성 반복.
    for task in tasks:
        direction = task["direction"]
        output_filename = task["filename"]
        
        # 현재 방향에 맞는 프롬프트 조립.
        # [공통규칙] + [현재 방향 지시] + [네거티브]를 결합하여 모델에 전달
        final_prompt = base_rules + "\n" + prompts_by_direction[direction] + "\n" + negative_prompt

        print(f"⏳ '{direction}' 측면 이미지 생성 중...")


        try:
            # 5. 모델 호출. (이미지 생성 요청)
            # contents 인자에 '레퍼런스 이미지'와 '텍스트 프롬프트'를 모두 전달하여 
            # Gemini의 이미지 참조 및 생성 능력을 활용.
            # Vertex AI Studio 설정: 온도 0.1 이하, 이미지 출력.
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    # 1번 이미지.(레퍼런스)
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/jpeg"  # 또는 image/png, 입력 파일에 맞춰 조정 가능.
                    ),
                    # 텍스트 프롬프트.
                    final_prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,  # 온도 설정 (0.1): 결과물의 일관성을 높이고 창의성을 낮추기.
                    # 모델이 이미지를 반환하도록 설정. (모델 스펙에 따라 파라미터가 다를 수 있음)
                    # 만약 순수 Imagen 모델이라면 generate_images 메서드를 써야 할 수도 있음.
                    # 여기서는 Gemini 멀티모달(입력:이미지+텍스트 -> 출력:이미지)을 가정.
                )
            )
            
            # 6. 응답 처리 및 이미지 저장.
            # Gemini 모델은 이미지 생성 결과를 response.parts 내의 inline_data로 반환
            if response.parts:
                for part in response.parts:
                    # 바이너리 데이터(이미지)가 있는지 확인.
                    if part.inline_data:
                        image_data = part.inline_data.data
                        
                        # 바이트 데이터를 이미지 파일로 저장.
                        img = Image.open(io.BytesIO(image_data))
                        filepath = os.path.join('apioutput',output_filename)
                        img.save(filepath)
                        print(f"   저장 완료: {filepath}")
                        break # 이미지를 하나 찾으면 저장하고 다음 태스크로.
                else:
                    # 루프가 break 없이 끝났다면 이미지가 없다는 뜻.
                    print(f"    경고: 모델 응답에 이미지 데이터가 없습니다. (텍스트 응답일 수 있음)")
                    print(f"   응답 내용: {response.text}")
            else:
                print(f"   오류: 모델로부터 응답이 비어있습니다.")

        except Exception as e:
            print(f"   '{direction}' 이미지 생성 중 에러 발생: {e}")

    print("\n 모든 추가 뷰 이미지 생성 작업이 끝났습니다.")
