import cv2
import numpy as np

# 비디오 파일 경로
video1_path = '0output.mp4'
video2_path = '1output.mp4'
video3_path = '2output.mp4'

# 비디오 캡처 객체 생성
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)
cap3 = cv2.VideoCapture(video3_path)

# 비디오 창 크기 설정
window_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
window_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 결과 화면 생성
result_width = window_width * 3  # 가로로 3개의 비디오를 나란히 표시하기 위해 가로 길이 확장
result = np.zeros((window_height, result_width, 3), dtype=np.uint8)

# 비디오 프레임 수 확인
frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count3 = int(cap3.get(cv2.CAP_PROP_FRAME_COUNT))

# 비디오 저장 설정
output_path = 'output_video.mp4'
output_fps = 10  # 저장할 비디오의 프레임 속도 설정
output_codec = cv2.VideoWriter_fourcc(*'mp4v')  # 비디오 코덱 설정
output_writer = cv2.VideoWriter(output_path, output_codec, output_fps, (result_width, window_height))

delay = 30

# 비디오 재생
while cap1.isOpened() or cap2.isOpened() or cap3.isOpened():
    # 비디오1 재생
    if cap1.isOpened():
        ret1, frame1 = cap1.read()
        if ret1:
            result[:, :window_width, :] = frame1  # 비디오1의 프레임을 결과 화면 왼쪽에 배치
        else:
            cap1.release()

    # 비디오2 재생
    if cap2.isOpened():
        ret2, frame2 = cap2.read()
        if ret2:
            result[:, window_width:window_width*2, :] = frame2  # 비디오2의 프레임을 결과 화면 가운데에 배치
        else:
            cap2.release()

    # 비디오3 재생
    if cap3.isOpened():
        ret3, frame3 = cap3.read()
        if ret3:
            result[:, window_width*2:, :] = frame3  # 비디오3의 프레임을 결과 화면 오른쪽에 배치
        else:
            cap3.release()

    # 결과 화면 표시
    cv2.imshow('Videos', result)

    # 비디오 저장
    output_writer.write(result)

    # 모든 비디오가 마지막 프레임에 도달한 경우 종료
    if (not cap1.isOpened() or cap1.get(cv2.CAP_PROP_POS_FRAMES) == frame_count1) and \
       (not cap2.isOpened() or cap2.get(cv2.CAP_PROP_POS_FRAMES) == frame_count2) and \
       (not cap3.isOpened() or cap3.get(cv2.CAP_PROP_POS_FRAMES) == frame_count3):
        break

    # 키 입력 대기
    if cv2.waitKey(delay) == ord('q'):
        break

# 비디오 저장 종료
output_writer.release()

# 비디오 재생 종료
cap1.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()