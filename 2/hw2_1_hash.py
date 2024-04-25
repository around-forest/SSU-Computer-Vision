import cv2
import numpy as np
import glob
import argparse

def find_building_in_image(target_image_path, building_image_path):
    # 이미지 로드
    target_image = cv2.imread(target_image_path)
    building_image = cv2.imread(building_image_path)

    # 빌딩 이미지를 16*16 평균 해쉬 변환
    building_gray = cv2.cvtColor(building_image, cv2.COLOR_BGR2GRAY)
    building_gray = cv2.resize(building_gray, (16, 16))
    building_avg = building_gray.mean()
    bi = 1 * (building_gray > building_avg)
    building_hash = bi
    print(building_hash)

    # 타깃 이미지를 16*16 평균 해쉬 변환
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.resize(target_gray, (16, 16))
    target_avg = target_gray.mean()
    ti = 1 * (target_gray > target_avg)
    target_hash = ti
    print(target_hash)

    # 해밍거리 측정
    building_hash = building_hash.reshape(1, -1)
    target_hash = target_hash.reshape(1, -1)
    distance = (building_hash != target_hash).sum()

    # 해밍거리 25%
    if distance / 256 < 0.25:
        print('TRUE')
        cv2.imshow("building_image", building_image)
        cv2.imshow("target_image", target_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        print(distance / 256)
        print('FALSE')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checkerboard cell count")
    parser.add_argument("image_path", help="checkerboard image file address")
    args = parser.parse_args()

    # 예시 이미지 경로 (이 부분은 실제 이미지 경로로 변경해야 함)
    target_image_path = args.image_path  # 대상 이미지 경로
    building_image_path = 'ESB_mask8.png'  # 엠파이어 스테이트 빌딩 이미지 경로

    # 함수 호출하여 결과 확인
    find_building_in_image(target_image_path, building_image_path)
