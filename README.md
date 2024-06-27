# Cải tiến YOLOv8 cho tác vụ nhận diện biển báo giao thông


## Kiến trúc YOLOv8 cải tiến
<p align="center">
  <img src="img\archi.jpg" width="1024" title="details">
</p>

<!-- ## Requirements -->


<!-- ## Environment
```
  pip install -r requirements.txt
``` -->

## Tập dữ liệu thực nghiệm
### Tải tập dữ liệu thực nghiệm
* Bạn có thể tải tập dữ liệu thực nghiệm từ Roboflow: [Traffic Sign Detection Image Dataset](https://universe.roboflow.com/trafficsigndetection-2u5ca/trafficsigndetection-gmvvi/dataset/7). 
Nguồn: [Vietnam-Traffic-Sign-Detection Computer Vision Project](https://universe.roboflow.com/vietnam-traffic-sign-detection/vietnam-traffic-sign-detection-2i2j8).

Ngoài ra, bạn cũng có thể tải tập dữ liệu trên từ Roboflow với đoạn code sau:
```
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="8BHje3hqCJV6zmRR7UQM")
project = rf.workspace("trafficsigndetection-2u5ca").project("trafficsigndetection-gmvvi")
version = project.version(7)
dataset = version.download("yolov8")
```

### Phân chia tập dữ liệu thực nghiệm

Bộ dữ liệu này được chia thành các tập:
- Tập huấn luyện: chiếm 60 % tập dữ liệu.
- Tập kiểm định: chiếm 20 % tập dữ liệu.
- Tập kiểm tra: chiếm 20 % tập dữ liệu.


Một số hình ảnh về tập dữ liệu thực nghiệm: 

<p align="center">
  <img src="img\dataa.jpg" width="1024" title="details">
</p>

## Phương pháp đề xuất
 Chúng tôi điều chỉnh kiến trúc mô hình YOLOv8 bằng thêm vào những phương pháp sau:
  - Một lớp nhận diện vật thể nhỏ có feature map là 160x160
  - **SPD-Conv mô đun**
  - **ResBlockCBAM mô đun**
  <p align="center">
  <img src="img\archi.jpg" width="1024" title="details">
</p>

## Kết quả thực nghiệm của mô hình trên tập dữ liệu
| Lớp nhận diện vật thể nhỏ | SPD | ResBlock_CBAM | Precision | Recall | mAP@0.5(%) | mAP@0.5:0.95(%) |
| :--: | :-: | :-: | :-: | :-: | :-: | :-: |
| | | |0.752 | 0.704 | 0.753 | 0.61 |
| | ✓ | |0.816 | 0.662 | 0.755 | 0.612|
| ✓ | | ✓ | 0.698 | 0.817 | 0.818 | 0.67|
| ✓ | ✓ | | 0.798 | 0.777 | 0.822 | 0.668 |
| ✓ | ✓ | ✓ | 0.774 | 0.76 | 0.832 | 0.681 |