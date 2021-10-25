# LabelImg

## 安裝
* 下載並安裝 [Python](https://www.python.org/)
* 至[原專案](https://github.com/tzutalin/labelImg)複製專案資料夾
* 依以下指令安裝相關依賴
```
git clone https://github.com/tzutalin/labelImg.git
cd labelImg/

pip install pyqt5 lxml
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py
```

## 製作資料集
1. 創建 dataset 資料夾及、子資料夾及所需檔案，其中 JPEGImages/ 存放欲標記的圖檔 Annotations/ 存放標記檔案，其餘的會在產生 YOLO 格式訓練檔案時會用到
```bash
dataset/
├─Annotations/
├─ImageSets/
│  └─Main/
│          test.txt
│          train.txt
│          trainval.txt
│          val.txt
└─JPEGImages/
```

2. 修改 labelImg/data/predefined_classes.txt 中的內容為欲標記的類別
```
class1
class2
class3
.
.
.
```
3. 打開 LabelImg 開啟欲標記的資料集的資料夾，包括圖檔資料夾及標記檔資料夾，並將標記格式設為 PascalVOC
<p float="left">
  <img src="./images/label/label_1.png" width="400" />
</p>

4. 框選欲標記的物件，選好類別後點儲存按鈕，標記檔就會輸出至 Annotations/，檔名會與 JPEGImages/ 中的圖片相對應
<p float="left">
  <img src="./images/label/label_2.png" width="400" />
</p>

5. LabelImg 快捷鍵可以參考[原專案](https://github.com/tzutalin/labelImg)


# TensorFlow YOLOv4
