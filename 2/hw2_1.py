import cv2
import numpy as np
import argparse

# RANSAC 원근 변환 근사 계산
def find_building_in_image(target_image_path, building_image_path):
    # 이미지 로드
    target_image = cv2.imread(target_image_path)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    building_image = cv2.imread(building_image_path)
    building_gray = cv2.cvtColor(building_image, cv2.COLOR_BGR2GRAY)

    target_gray = cv2.Canny(target_gray, 100, 200)
    building_gray = cv2.Canny(building_gray, 100, 200)

    # ORB 특징점 알고리즘 객체 생성
    orb = cv2.ORB_create()

    # KAZE 특징점 알고리즘 객체 생성
    kaze = cv2.KAZE_create()

    # 특징점과 서술자 추출
    keypoints1, descriptors1 = kaze.detectAndCompute(building_gray, None)
    keypoints2, descriptors2 = kaze.detectAndCompute(target_gray, None)

    # BF-Hamming Matcher 객체 생성 for Hamming
    hamming = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

    # BF Matcher 객체 생성 for KAZE
    bf = cv2.BFMatcher_create()

    # KnnMatch 매칭
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 매칭점 그리기
    all_matches = cv2.drawMatches(building_image, keypoints1, target_image, keypoints2,
                                  matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    # 원근 변환 및 영역 표시
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # RANSAC 변환 행렬 근사 계산
    matrix, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC,
                                      ransacReprojThreshold=5.0, maxIters=2000)
    h, w = building_image.shape[:2]
    pts = np.float32([[[0,0]], [[0,h-1]], [[w-1,h-1]], [[w-1,0]]])
    dst = cv2.perspectiveTransform(pts, matrix)
    target_image = cv2.polylines(target_image, [np.int32(dst)], True, 255, 3,
                                   cv2.LINE_AA)

    # 정상치 매칭 그리기
    matchesMask = mask.ravel().tolist()
    good_matches = cv2.drawMatches(building_image, keypoints1, target_image, keypoints2,
                                  matches, None, matchesMask=matchesMask,
                                  flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    # 모든 매칭점과 정상치 비율
    accuracy = float(mask.sum()) / mask.size
    print(accuracy)

    cv2.imshow("all matches", all_matches)
    cv2.imshow("good matches", good_matches)
    cv2.waitKey()
    cv2.destroyAllWindows()


    # 매칭 결과에 따라 True 또는 False 반환
    # 여기서는 매칭된 특징점이 임계값(예: 10)보다 많으면 True 반환
    #return len(matches) > 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checkerboard cell count")
    parser.add_argument("image_path", help="checkerboard image file address")
    args = parser.parse_args()

    # 예시 이미지 경로 (이 부분은 실제 이미지 경로로 변경해야 함)
    target_image_path = args.image_path  # 대상 이미지 경로
    building_image_path = 'ESB_mask8.png'  # 엠파이어 스테이트 빌딩 이미지 경로

    # 함수 호출하여 결과 확인
    find_building_in_image(target_image_path, building_image_path)
    
