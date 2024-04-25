import cv2
import argparse # 인자 처리
import numpy as np

def find_checkerboard_corners(image_path):
    src = cv2.imread(image_path)
    dst = src.copy()

    if src is None:
        print('Cannot load image')
        return
    
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    min_distance = 10
    max_corners = 0
    quality_level = 0.3

    corners = cv2.goodFeaturesToTrack(blur, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)

    if np.sqrt(len(corners)) - 1 <= 9:
        print('8 x 8')
    else:
        print('10 x 10')

    if corners is not None:
        for i in range(corners.shape[0]): # 코너 갯수만큼 반복문
            pt = (int(corners[i, 0, 0]), int(corners[i, 0, 1])) # x, y 좌표 받아오기
            cv2.circle(dst, pt, 5, (0, 0, 255), 2) # 받아온 위치에 원

    else:
        print('Cannot find checkerboard')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checkerboard cell count")
    parser.add_argument("image_path", help="checkerboard image file address")
    args = parser.parse_args()

    find_checkerboard_corners(args.image_path)
    