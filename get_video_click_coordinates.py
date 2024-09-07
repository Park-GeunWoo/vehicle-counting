import cv2

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭 이벤트
        print(f"Clicked at ({x}, {y})")
        # 클릭한 좌표를 이미지에 표시
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"({x},{y})", (x, y), font, 0.5, (255, 0, 0), 2)
        cv2.imshow('Image', img)

video_path = "C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/3_Input 남측_G88/16-19/alwa_20240529_185759_F.mp4" #1
# video_path="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/8_Output 북측_G96/16-19/alwa_20240529_164029_F.mp4" #2
# video_path="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/7_Output 남측_G95/16-19/alwa_20240529_181815_F.mp4" #3
# video_path="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/6_Output 서측_G89/16-19/alwa_20240529_162522_F.mp4"
# video_path="C:/Users/USER/Downloadss/20240529_서평택IC사거리 교통량조사(76G)/5_Output 동측_G91/16-19/alwa_20240529_181533_F.mp4" #4
# video_path="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/2_Input 서측_G93/16-19/alwa_20240529_180041_F.mp4"
# 영상 파일 열기
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

ret, img = cap.read()
if not ret:
    print("Error: Could not read the video frame.")
    cap.release()
    exit()

cv2.imshow('Image', img)

cv2.setMouseCallback('Image', click_event)

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
