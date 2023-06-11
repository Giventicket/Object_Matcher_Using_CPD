
import cv2
import numpy as np

# 비디오 파일 경로 리스트
video_paths = ["video/output_order1_0.mp4", "video/output_order1_1.mp4", "video/output_order1_2.mp4"]

# 비디오 캡처 객체 생성
video_captures = [cv2.VideoCapture(path) for path in video_paths]
import pdb;pdb.set_trace()
# 비디오 크기 가져오기
widths = []
heights = []
for capture in video_captures:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    widths.append(width)
    heights.append(height)

# 연결된 비디오의 총 가로 길이와 최대 세로 길이 계산
total_width = sum(widths)
max_height = max(heights)

# 출력 비디오 작성자 생성
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 사용할 코덱 설정
out = cv2.VideoWriter("output.mp4", fourcc, 10.0, (total_width, max_height))  # 출력 파일명, 코덱, FPS, 크기 설정

# 프레임 읽어와서 가로로 이어붙여 출력 비디오에 작성
while True:
    frames = []
    for capture in video_captures:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    if len(frames) == 0:
        break
    combined_frame = np.concatenate(frames, axis=1)
    out.write(combined_frame)

# 작업 완료 후 객체 해제
out.release()
for capture in video_captures:
    capture.release()



##################################################################################################################################


# import cv2
# import os

# # 폴더 경로
# folder_path = "folder/"

# # 폴더 내 파일 목록 가져오기
# file_list = os.listdir(folder_path)

# # 파일 이름을 정렬하여 숫자로 오름차순 정렬
# file_list = sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))

# # 가장 긴 영상의 길이 확인
# max_duration = 0
# for file_name in file_list:
#     import pdb;pdb.set_trace()
#     video = cv2.VideoCapture(folder_path + file_name)
#     duration = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#     if duration > max_duration:
#         max_duration = duration
#     video.release()

# # 첫 번째 영상을 기준으로 가로, 세로 크기 가져오기
# first_video = cv2.VideoCapture(folder_path + file_list[0])
# height, width = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
# fps = first_video.get(cv2.CAP_PROP_FPS)
# first_video.release()

# # 비디오 작성자 생성
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 사용할 코덱 설정
# out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))  # 출력 파일명, 코덱, FPS, 크기 설정

# # 영상을 가장 긴 영상의 길이로 맞추어 저장
# for file_name in file_list:
#     video = cv2.VideoCapture(folder_path + file_name)
#     duration = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     # 짧은 영상의 경우 마지막 프레임을 연장
#     if duration < max_duration:
#         last_frame = None
#         while True:
#             ret, frame = video.read()
#             if not ret:
#                 break
#             last_frame = frame
        
#         for _ in range(max_duration - duration):
#             out.write(last_frame)
    
#     # 영상을 출력 파일에 저장
#     while True:
#         ret, frame = video.read()
#         if not ret:
#             break
#         out.write(frame)
    
#     video.release()

# # 작업 완료 후 객체 해제
# out.release()

################################################################################################################################

# import cv2

# def resize_video(video, target_width, target_height):
#     # 현재 프레임의 크기를 가져옴
#     height, width, _ = video.shape

#     # 타겟 크기와 현재 크기의 비율을 계산
#     width_ratio = target_width / width
#     height_ratio = target_height / height

#     # 비율에 따라 크기를 조정하여 리사이징
#     resized_video = cv2.resize(video, (0, 0), fx=width_ratio, fy=height_ratio)

#     return resized_video

# def extend_video(video, target_length):
#     # 현재 프레임 수와 타겟 길이의 비율을 계산
#     current_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#     ratio = target_length / current_length

#     # 비율에 따라 마지막 프레임을 연장하여 확장된 비디오 생성
#     last_frame = video[-1]
#     extended_video = [last_frame] * (target_length - current_length)
#     extended_video = np.concatenate((video, extended_video))

#     return extended_video

# # 영상 파일 경로
# video1_path = "video/test/output_order1_0.mp4"
# video2_path = "video/test/output_order1_1.mp4"
# video3_path = "video/test/output_order1_2.mp4"

# # 영상 로드
# video1 = cv2.VideoCapture(video1_path)
# video2 = cv2.VideoCapture(video2_path)
# video3 = cv2.VideoCapture(video3_path)

# # 가장 긴 영상의 길이 가져오기
# max_length = max(
#     int(video1.get(cv2.CAP_PROP_FRAME_COUNT)),
#     int(video2.get(cv2.CAP_PROP_FRAME_COUNT)),
#     int(video3.get(cv2.CAP_PROP_FRAME_COUNT))
# )

# # 영상 확장
# video1_extended = extend_video(video1, max_length)
# video2_extended = extend_video(video2, max_length)
# video3_extended = extend_video(video3, max_length)

# # 영상 크기 조정
# target_width = 640
# target_height = 480

# video1_resized = resize_video(video1_extended, target_width, target_height)
# video2_resized = resize_video(video2_extended, target_width, target_height)
# video3_resized = resize_video(video3_extended, target_width, target_height)

# # 영상 재생
# while True:
#     # 각 영상에서 프레임 읽기
#     ret1, frame1 = video1_resized.read()
#     ret2, frame2 = video2_resized.read()
#     ret3, frame3 = video3_resized.read()

#     # 모든 영상의 프레임을 읽으면 종료
#     if not ret1 or not ret2 or not ret3:
#         break

#     # 영상을 화면에 표시
#     cv2.imshow("Video 1", frame1)
#     cv2.imshow("Video 2", frame2)
#     cv2.imshow("Video 3", frame3)

#     # 'q' 키를 누르면 종료
#     if cv2.waitKey(1) == ord('q'):
#         break

# # 영상 재생 종료 후, 창 닫기
# cv2.destroyAllWindows()

# # 영상 파일 해제
# video1.release()
# video2.release()
# video3.release()