import init_path
from orion.utils.misc_utils import *

def add_text_on_image(img, text, pos=(10,30), color=(255,255,255), fontsize=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = fontsize
    fontColor = color
    lineType = 2
    cv2.putText(img, text, 
        pos, 
        font, 
        fontScale,
        fontColor,
        lineType)
    return img

video_writer = VideoWriter('./annotations/human_demo/1294_demo', video_name=f"stitched_sim_res.mp4", fps=30)

# read all frames from a video
video_capture = cv2.VideoCapture('annotations/human_demo/1294_demo/sim_output_60.mp4')
video_seq1 = []
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    video_seq1.append(frame)

video_capture = cv2.VideoCapture('./annotations/human_demo/1294_demo/sim_output_600.mp4')
video_seq2 = []
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    video_seq2.append(frame)

assert len(video_seq1) == len(video_seq2)
print("length of video_seq1:", len(video_seq1))

for i in range(len(video_seq1)):
    image_frame = []

    video_seq1[i] = add_text_on_image(video_seq1[i], "60", pos=(100, 600), color=(255, 0, 0), fontsize=1.5)
    video_seq2[i] = add_text_on_image(video_seq2[i], "600 (smoothed)", pos=(100, 600), color=(255, 0, 0), fontsize=1.5)
    image_frame.append(video_seq1[i])
    image_frame.append(video_seq2[i])
    
    image_frame = np.concatenate(image_frame, axis=1)
    video_writer.append_image(image_frame)
video_writer.save(bgr=True)