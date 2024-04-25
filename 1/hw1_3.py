import cv2
import argparse
import numpy as np

def order_points(pts):
    # 좌상단, 우상단, 우하단, 좌하단 순서로 정렬
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def auto_canny(image, sigma=0.33):
    # Canny 엣지 검출을 위한 자동 임계값 계산
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def find_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = auto_canny(blurred)
    
    # 엣지에서 윤곽선 찾기
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # 윤곽선을 순회하며 체커보드의 네 모서리를 찾기
    for c in contours:
        # 윤곽선 근사화
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # 정사각형의 네 모서리를 찾았는지 확인
        if len(approx) == 4:
            return order_points(approx.reshape(4, 2))
    return None

def transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    """
    #변환될 이미지의 너비와 높이 계산
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(( (tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    """
    maxWidth = 500
    maxHeight = 500

    # 목표 좌표 배열
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # perspective transform을 위한 행렬 계산
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checkerboard cell count")
    parser.add_argument("image_path", help="checkerboard image file address")
    args = parser.parse_args()

    # 이미지 로드
    image = cv2.imread(args.image_path)
    corners = find_corners(image)

    if corners is not None:
        warped = transform(image, corners)
        cv2.imshow('Warped', warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("체커보드 모서리를 찾을 수 없습니다.")