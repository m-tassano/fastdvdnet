import cv2
import os

video_path = 'dataset/videoplayback.mp4'
save_path = 'dataset/video1'
frame_skip = 100
cap = cv2.VideoCapture(video_path)

if not os.path.exists(save_path):
    os.makedirs(save_path)
else:
    print("Directory already exists")
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame_count/ frame_rate
print("Frame count = ", frame_count)
print("Video Duration = ", duration)
print("Frame Rate = ", frame_rate)
count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    if not count % frame_skip == 0:
        count +=1
        continue

    # Save the frame as an image
    cv2.imwrite(f'{save_path}/frame{count}.jpg', frame)

    count += 1
    print("Frame: ", count, "/", frame_count)

print(count)
cap.release()
cv2.destroyAllWindows()