import cv2
import os

# 폴더 경로
folder_list = ["1","2","3"]

for j in range(3) :
    folder_path = "worst/" + str(folder_list[j])
    # 폴더 내 파일 목록 가져오기
    file_list = os.listdir(folder_path)
    # 파일 이름을 정렬하여 숫자로 오름차순 정렬
    file_list = sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))
    print(file_list)
    # 첫 번째 이미지 파일을 기준으로 가로, 세로 크기 가져오기
    first_image = cv2.imread(folder_path + "/" + file_list[0])
    height, width, _ = first_image.shape

    # 비디오 작성자 생성
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 사용할 코덱 설정
    # import pdb;pdb.set_trace()
    out = cv2.VideoWriter(str(j) + "output.mp4", fourcc, 10.0, (width, height))  # 출력 파일명, 코덱, FPS, 크기 설정

    # 이미지를 영상으로 변환하여 저장
    for file_name in file_list:
        image = cv2.imread(folder_path + "/" + file_name)
        out.write(image)

    # 작업 완료 후 객체 해제
    out.release()
