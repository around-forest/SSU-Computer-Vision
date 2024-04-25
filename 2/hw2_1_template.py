# 템플릿 매칭으로 객체 위치 검출 (template_matching.py)

import cv2
import numpy as np

def apply_canny(image, sigma=0.33):
    # Canny 에지 감지를 위한 자동 임계값 계산
    median = np.median(image)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edged = cv2.Canny(image, lower, upper)
    return edged

# 입력이미지와 템플릿 이미지 읽기
img = cv2.imread('6.jpg')
template = cv2.imread('ESB_mask8.png')
th, tw = template.shape[:2]

# 에지 감지 적용
img_edges = apply_canny(img)
template_edges = apply_canny(template)

# 대상 이미지가 템플릿보다 작은 경우 크기 조정
if img_edges.shape[0] < th or img_edges.shape[1] < tw:
    scale_height = th / img_edges.shape[0]
    scale_width = tw / img_edges.shape[1]
    scale = max(scale_height, scale_width)

    new_height = int(img_edges.shape[0] * scale)
    new_width = int(img_edges.shape[1] * scale)
    img_blur = cv2.resize(img_edges, (new_width, new_height), interpolation=cv2.INTER_AREA)

# 3가지 매칭 메서드 순회
methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', \
                                     'cv2.TM_SQDIFF_NORMED']

# 엠파이어 스테이트 빌딩 검출 여부
building_detected = False

for i, method_name in enumerate(methods):
    img_draw = img.copy()
    method = eval(method_name)

    # 템플릿 매칭   ---①
    res = cv2.matchTemplate(img_edges, template_edges, method)

    # 최솟값, 최댓값과 그 좌표 구하기 ---②
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(method_name, min_val, max_val, min_loc, max_loc)

    # 임계값 설정
    threshold = 0.8

    # TM_SQDIFF의 경우 최솟값이 좋은 매칭, 나머지는 그 반대 ---③
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        match_val = min_val
        building_detected = match_val < threshold
    else:
        top_left = max_loc
        match_val = max_val
        building_detected = match_val > threshold

    # 매칭 좌표 구해서 사각형 표시   ---④      
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    cv2.rectangle(img_draw, top_left, bottom_right, (0,0,255),2)

    # 매칭 포인트 표시 ---⑤
    cv2.putText(img_draw, str(match_val), top_left, \
                cv2.FONT_HERSHEY_PLAIN, 2,(0,255,0), 1, cv2.LINE_AA)
    cv2.imshow(method_name, img_draw)

print(building_detected)
cv2.waitKey(0)
cv2.destroyAllWindows()