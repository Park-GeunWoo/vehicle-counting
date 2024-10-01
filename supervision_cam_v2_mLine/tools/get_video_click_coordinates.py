import cv2

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭 이벤트
        print(f"{x}, {y}")
        # 클릭한 좌표를 이미지에 표시
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"({x},{y})", (x, y), font, 0.5, (255, 0, 0), 2)
        cv2.imshow('Image', img)



video_path="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/1_Input 동측_G87/16-19/alwa_20240529_185838_F.mp4"#8
video_path="C:/Users/USER/Desktop/alwa_20240529_185838_F.mp4"

# 영상 파일 열기
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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
