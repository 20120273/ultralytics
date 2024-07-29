import gradio as gr
import PIL.Image as Image
import sys
import cv2
sys.path.insert(0, 'D:/20120273_20120516/Code/ultralytics')

from ultralytics import ASSETS, YOLO

model = YOLO("D:/20120273_20120516/Code/bestyolo.pt")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


video_path = "path/to/video.mp4"
def XuLy(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))  # Get video width
    frame_height = int(cap.get(4))  # Get video height
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
    frame_size = (frame_width, frame_height)  # Determine the size of video frames
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Define codec
    output_video = "output_recorded.mp4"
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            out.write(annotated_frame) 
            yield cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), None
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    yield None, output_video
input_video = gr.Video(sources=None, label="Input Video")
output_frame = gr.Image(type="numpy", label="Output Frames")
output_video_file = gr.Video(label="Output video")

interface_video = gr.Interface(
    fn=XuLy,
    inputs=input_video,
    outputs= [output_frame, output_video_file]
)

if __name__ == "__main__":
    # iface.launch(debug = True,share = True)
    # gr.TabbedInterface(
    # [interface_image, interface_video],
    # tab_names=['Image inference', 'Video inference']).queue().launch()
    # result = model.track(source='',save = True, show = True,tracker="bytetrack.yaml")
    
    interface_video.launch(debug=True)