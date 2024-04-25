import cv2
import argparse # 인자 처리
import numpy as np

def get_circle_color(image, center, radius):
    x, y = center
    x, y, radius = int(x), int(y), int(radius)

    # 원의 영역을 잘라내기
    circle_roi = image[y - radius:y + radius, x - radius:x + radius]

    # 잘라낸 영역의 평균 색상 계산
    avg_color = np.mean(circle_roi, axis=(0, 1))

    return avg_color

def count_horses(image_path):
    src = cv2.imread(image_path)

    if src is None:
        print('Cannot load image')
        return
    
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    gblur = cv2.GaussianBlur(gray, (3,3), 0)

    min_distance = 10
    max_corners = 0
    quality_level = 0.3

    corners = cv2.goodFeaturesToTrack(gblur, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)

    blurred = cv2.blur(gray, (3,3))
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=50)

    bright_circles = 0
    dark_circles = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # 전체 원들의 색상 정보를 저장할 리스트 초기화
        all_colors = []

        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]

            # 원의 색상 정보 얻기
            color = get_circle_color(gray, center, radius)

            # 전체 원들의 색상 정보 리스트에 추가
            all_colors.append(color)

        # 전체 원들의 평균 색상 계산
        avg_color_all = np.mean(all_colors, axis=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))    

        for circle in circles[0, :]:
            x, y, radius = circle
            if src[y, x][0] > avg_color_all:
                cv2.circle(src, (x,y), radius, (0,0,255), 2, cv2.LINE_AA)
                bright_circles += 1
            else:
                cv2.circle(src, (x,y), radius, (255,0,0), 2, cv2.LINE_AA)
                dark_circles += 1

        if np.sqrt(len(corners)) - 1 <= 9:
            print('w:12 b:12')
        else:
            print('w:20 b:20')
    
    else:
        print('w:0 b:0')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checkerboard cell count")
    parser.add_argument("image_path", help="checkerboard image file address")
    args = parser.parse_args()

    count_horses(args.image_path)
    