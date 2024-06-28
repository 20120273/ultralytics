# Cải tiến YOLOv8 cho tác vụ nhận diện biển báo giao thông


## Giới thiệu
Dự án này tập trung vào việc cải tiến kiến trúc YOLOv8 cho việc nhận diện biển báo giao thông ở Việt Nam.

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

Cấu trúc tập dữ liệu khi tải về: 

       TrafficSignDetection Image Dataset
          └── data   
               ├── data.yaml
               ├── README.dataset.txt
               ├── README.roboflow.txt
               ├── train
               │    ├── images
               │    │    ├── train_img1.png
               │    │    └── ...
               │    ├── labels
               │    │    ├── train_annotation1.txt
               │    │    └── ...    
               ├── valid
               │    ├── images
               │    │    ├── valid_img1.png
               │    │    └── ...
               │    ├── labels
               │    │    ├── valid_annotation1.txt
               │    │    └── ... 
               ├── test
               │    ├── images
               │    │    ├── test_img1.png
               │    │    └── ...
               │    ├── labels
               │    │    ├── test_annotation1.txt
               │    │    └── ... 

Một số hình ảnh về tập dữ liệu thực nghiệm: 

<p align="center">
  <img src="img\dataa.jpg" width="1024" title="details">
</p>

## Phương pháp đề xuất
 Chúng tôi điều chỉnh kiến trúc mô hình YOLOv8 bằng thêm vào những phương pháp sau:
  - **Một lớp nhận diện vật thể nhỏ**: Được thêm vào để nâng cao khả năng phát hiện các đối tượng nhỏ như biển báo giao thông, với bản đồ đặc trưng 160x160.
  - **SPD-Conv mô đun**:  Giúp cải thiện quá trình trích xuất đặc trưng bằng cách tăng cường khả năng phân tích không gian.
  - **ResBlockCBAM mô đun**: Tích hợp để cải thiện biểu diễn đặc trưng, giúp mô hình tập trung vào các khu vực quan trọng trong ảnh.

## Kiến trúc YOLOv8 cải tiến
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

## Một số hình ảnh của mô hình YOLOv8 trong tác vụ nhận diện biển báo giao thông với tập dữ liệu thực nghiệm

<p align="center">
  <img src="img\3.jpg" width="1024" title="details">
</p>


<p align="center">
  <img src="img\2.jpg" width="1024" title="details">
</p>


<p align="center">
  <img src="img\1.jpg" width="1024" title="details">
</p>