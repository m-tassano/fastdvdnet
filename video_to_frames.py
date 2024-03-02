import cv2, os
video_path = 'data_set/videoplayback.mp4'
if not os.path.exists(video_path):
    print("Video does not exist")
else:
    print("Found video")
save_path = 'data_set/video1'
if os.path.exists(save_path):
    print("Path already exists")
    quit()
os.mkdir(save_path)
cap = cv2.VideoCapture(video_path)

if (cap.isOpened()== False):
    print("Error opening video file")
cnt = 1
# Read until video is completed
while(cap.isOpened()):
      
# Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
    # Display the resulting frame
        # cv2.imshow('Frame', frame)
        cv2.imwrite(f'{save_path}/{cnt}.jpg', frame)
        cnt+=1
    # Press Q on keyboard to exit
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

  
# Break the loop
    else:
        break
  
# When everything done, release
# the video capture object
cap.release()
  
# Closes all the frames
cv2.destroyAllWindows()