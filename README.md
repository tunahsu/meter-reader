# 類比電表辨識
<p float="left">
  <img src="./results/result4.png" width="400" />
  <img src="./results/result6.png" width="400" />
</p>

## 簡介
* 使用 TensorFlow 版的 YOLOv4 訓練模型
* 物件辨識的種類為數值、方形電表、圓形電表
* 主要以極坐標轉換來獲取影像中的刻度、指針
* 計算刻度、指針中心坐標用以推算其距離關係
* 輸入圖片可以辨識類比電表中指針指到的數值

## 套件
```
tensorflow-gpu==2.3.0rc0
opencv-python
lxml
tqdm
absl-py
matplotlib
easydict
pillow
pytesseract
```

## 下載
請下載最新版，將訓練過的模型解壓至專案根目錄並將資料夾命名為 checkpoints/，資料集解壓至 data/
* [Dataset](https://drive.google.com/drive/folders/17GL5mKnMv6qhyP1jZOL7Ub5QfbKR1hHR?usp=sharing)
* [Model](https://drive.google.com/drive/folders/1YPWNaVrifHLVj_yAzJmJnffDFmgr4hQD?usp=sharing)

## 測試
可使用資料集中得圖片作為測試資料
```
python detect.py --weights checkpoints\yolov4-analog-ammeter-160 --image test.jpg
```

## 模型訓練
先運行 sctipts/voc/ 中的腳本產生 YOLO 的資料集格式，請確保腳本中路徑是正確的，遷移學習的預訓練模型可至此下載 [yolov4.weights](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT)
```
meter-reader                                                                     
├─data                                                                           
│  ├─classes                                                                  
│  │      analog_ammeter.names
│  └─dataset                                                                  
│          analog_ammeter_train.txt                                           
│          analog_ammeter_val.txt         
```

```bash
cd scripts/voc/
python voc_gen_imagesets.py
python voc_convert.py
python voc_make_names.py
cd ../../

# 從頭開始訓練
# core/config.py 中 FISRT_STAGE_EPOCHS = 0 
python train.py

# 遷移學習
python train.py --weights ./data/yolov4.weights
```

## 遭遇問題及困難
* 目前使用的 OCR library(pytesseract) 效果不甚理想，數字辨識錯誤會嚴重影響最終的結果
* 對電表中 pattern 處理的演算法彈性不大，仍須對不同型態的電表類別去設計
* 幾乎沒有真實場景中的電表資料，很難估計在光影等因素的影響下成效如何
* 些許情況下無法辨識，如有兩圈刻度的圓形電表，或是其他非常規顯示的電表


## 預計增加功能
* [ ] 將 python 程式碼轉至 Android
* [ ] 使用更精確的數字 OCR library
* [ ] 增加其他類型電表的辨識功能
* [ ] 即時偵測/辨識
* [ ] 輸入影像角度自動校正