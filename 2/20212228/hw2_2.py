import cv2
import numpy as np
import argparse

def apply_canny(image):
    # Canny 에지 감지 적용
    return cv2.Canny(image, 100, 200)

def find_building_in_image(target_image_path, building_image_path):
    # 템플릿(엠파이어 스테이트 빌딩의 윗부분) 로드 및 에지 감지 적용
    template_image = cv2.imread(building_image_path)
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    template_edges = apply_canny(template_gray)

    # 대상 이미지 로드 및 에지 감지 적용
    target_image = cv2.imread(target_image_path)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    target_edges = apply_canny(target_gray)

    # 템플릿의 크기 보다 대상 이미지의 크기가 작을 경우 사이즈 조정
    th, tw = template_image.shape[:2]

    if target_edges.shape[0] < th or target_edges.shape[1] < tw:
        scale_height = th / target_edges.shape[0]
        scale_width = tw / target_edges.shape[1]
        scale = max(scale_height, scale_width)

        new_height = int(target_edges.shape[0] * scale)
        new_width = int(target_edges.shape[1] * scale)
        target_edges = cv2.resize(target_edges, (new_width, new_height), interpolation=cv2.INTER_AREA)


    # 템플릿 매칭 수행
    res = cv2.matchTemplate(target_edges, template_edges, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 매칭 결과 분석 및 출력
    max_threshold = 0.18
    if max_val > max_threshold:
        top_left = max_loc
        bottom_right = (top_left[0] + template_edges.shape[1], top_left[1] + template_edges.shape[0])
        cv2.rectangle(target_image, top_left, bottom_right, (0, 0, 255), 2)
        print("True")
        cv2.imshow("Detected", target_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("False")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checkerboard cell count")
    parser.add_argument("image_path", help="checkerboard image file address")
    args = parser.parse_args()

    # 예시 이미지 경로 (이 부분은 실제 이미지 경로로 변경해야 함)
    target_image_path = args.image_path  # 대상 이미지 경로
    building_image_path = 'ESB_mask8.png'  # 엠파이어 스테이트 빌딩 이미지 경로

    # 함수 호출하여 결과 확인
    find_building_in_image(target_image_path, building_image_path)


