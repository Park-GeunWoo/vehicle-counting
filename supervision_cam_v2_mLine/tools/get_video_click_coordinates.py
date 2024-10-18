import cv2
import os
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"{x}, {y}")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"({x},{y})", (x, y), font, 0.5, (255, 0, 0), 2)
        cv2.imshow('Image', img)



video_path = "input북측18.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Could not open video file.")
    exit()

ret, img = cap.read()

if not ret:
    print("Could not read the video frame.")
    cap.release()
    exit()

cv2.imshow('Image', img)

cv2.setMouseCallback('Image', click_event)

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
