import gradio as gr
import PIL.Image as Image
import sys
import cv2
sys.path.insert(0, 'D:/20120273_20120516/Code/ultralytics')

from ultralytics import ASSETS, YOLO

model = YOLO("D:/20120273_20120516/Code/bestyolo.pt")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def predict_image(img, conf_threshold, iou_threshold):
    """Predicts objects in an image using a YOLOv8 model with adjustable confidence and IOU thresholds."""
    results = model.track(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

        # Display the annotated frame
    #im =  cv2.imshow("YOLOv8 Tracking", annotated_frame)

    return im



def show_preds_image(image_path):
    image = cv2.imread(image_path)
    outputs = model.predict(source=image_path)
    results = outputs[0].cpu().numpy()
    for i, det in enumerate(results.boxes.xyxy):
        cv2.rectangle(
            image,
            (int(det[0]), int(det[1])),
            (int(det[2]), int(det[3])),
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
inputs_image = [
    gr.Image(type="filepath", label="Input Image"),
]
outputs_image = [
    gr.Image(type="numpy", label="Output Image"),
]
interface_image = gr.Interface(
    fn=show_preds_image,
    inputs=inputs_image,
    outputs=outputs_image,
    title="Mô hình nhận diện biển báo giao thông",
    description="Mô hình nhận diện biển báo giao thông Việt Nam dựa trên cải tiến mô hình YOLOv8.",
    cache_examples=False,
)


def show_preds_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame_copy = frame.copy()
            outputs = model.predict(source=frame)
            results = outputs[0].cpu().numpy()
            for i, det in enumerate(results.boxes.xyxy):
                cv2.rectangle(
                    frame_copy,
                    (int(det[0]), int(det[1])),
                    (int(det[2]), int(det[3])),
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
            yield cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

inputs_video = [
    gr.Video( label="Input Video"),

]
outputs_video = [
    gr.Image(type='numpy', label="Output Image"),
    #gr.components.Image(type="numpy", label="Output Image"),
]
interface_video = gr.Interface(
    fn=show_preds_video,
    inputs=inputs_video,
    outputs=outputs_video,
    title="Mô hình nhận diện biển báo giao thông",
    description="Mô hình nhận diện biển báo giao thông Việt Nam dựa trên cải tiến mô hình YOLOv8.",
    cache_examples=False,
)



iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Video( label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),

    ],
    outputs=[gr.PlayableVideo( label="Result")],
    title="Mô hình nhận diện biển báo giao thông",
    description="Mô hình nhận diện biển báo giao thông Việt Nam dựa trên cải tiến mô hình YOLOv8.",

)

if __name__ == "__main__":
    # iface.launch(debug = True,share = True)
    # gr.TabbedInterface(
    # [interface_image, interface_video],
    # tab_names=['Image inference', 'Video inference']).queue().launch()
    # result = model.track(source='',save = True, show = True,tracker="bytetrack.yaml")
    video_path = "D:/20120273_20120516/Code/ultralytics/ad.mp4"
    cap = cv2.VideoCapture(video_path)

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

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()