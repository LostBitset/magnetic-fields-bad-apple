import cv2

# Credit: https://github.com/CalvinLoke/bad-apple
# (applies to this file)

video_path = "BadApple.mp4"
cap = cv2.VideoCapture(video_path)
current_frame = 1
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while current_frame < total_frames:
    ret, frame = cap.read()
    frame_name = r"frames/" + "BadApple_" + str(current_frame) + ".jpg"
    cv2.imwrite(frame_name, frame)
    print(f"Wrote frame {frame_name}.")
    current_frame += 1
cap.release()

