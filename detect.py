# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync

# My imports
import pytesseract
import numpy as np
from operator import itemgetter
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


# 顯示圖片
def show_img(title, img):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 獲取數值物件中的字串
def digit_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789').strip()
    # show_img('Thresh', thresh)
    # print(text)
    return(text)


def get_center(img, result, obj):
    # 擷取物件區域
    img = img[obj['coord'][1]:obj['coord'][3], obj['coord'][0]:obj['coord'][2]]

    if obj['class'] == 1:
        height, width, _ = img.shape
        center_coord = [int(obj['coord'][0] + width * 0.75), int(obj['coord'][1] + height * 0.75)]

    elif obj['class'] == 2:
        height, width, _ = img.shape
        center_coord = [int(obj['coord'][0] + width * 0.5), int(obj['coord'][1] + height * 1.2)]

    elif obj['class'] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=200, minRadius=100, maxRadius=600)
        circles = np.uint16(np.around(circles))
        circle = circles[0][0]

        # 座標校正
        center_coord = [obj['coord'][0] + circle[0], obj['coord'][1] + circle[1]]
        center_radius = circle[2]
        # 畫圓
        cv2.circle(result,(center_coord[0], center_coord[1]), center_radius, (0, 255, 0), 2)

    # 畫圓心
    cv2.circle(result, (center_coord[0], center_coord[1]), 2, (0, 0, 255), 2)
    return center_coord


def get_pointer(img, result, obj, center_coord):
    img = img[obj['coord'][1]:obj['coord'][3], obj['coord'][0]:obj['coord'][2]]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    guassian = cv2.GaussianBlur(gray, (9, 9), 0)
    ret, thresh = cv2.threshold(guassian, 150, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(guassian, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 15)
    edges = cv2.Canny(thresh, 100, 150)

    # show_img('Thresh', thresh)
    # show_img('Edges', edges)

    # 找直線
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, lines=np.array([]), minLineLength=50, maxLineGap=5)
    
    for x1, y1, x2, y2 in lines[0]:
        # 因處理圖片只取電表區塊 座標需再作補正
        x1 += obj['coord'][0]
        y1 += obj['coord'][1]
        x2 += obj['coord'][0]
        y2 += obj['coord'][1]

        # 取離圓心遠的點
        pt1 = [x1, y1]
        pt2 = [x2, y2]
        diff1 = ((pt1[0] - center_coord[0]) ** 2 + (pt1[1] - center_coord[1]) ** 2) ** 0.5
        diff2 = ((pt2[0] - center_coord[0]) ** 2 + (pt2[1] - center_coord[1]) ** 2) ** 0.5
        
        pt1 = pt1 if diff1 > diff2 else pt2
        pt2 =  center_coord

        # 計算角度
        angle = int((np.degrees(np.arctan2((pt1[1] - pt2[1]), (pt1[0] - pt2[0]))) + 270) % 360)

        # 畫線
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return angle 


def meter_read(img, obj_list):  
    for obj in obj_list:
        if obj['class'] in [1, 2, 3, 4]: # 若物件類別為電表
            result = img.copy()

            # 獲取指針、圓
            center_coord = get_center(img, result, obj)
            pointer_angle = get_pointer(img, result, obj, center_coord)

            # 只取一個電表
            break

    # 獲取每個數值相對應的角度
    value_list = []
    for obj in obj_list:
        if obj['class'] == 0: # 若物件類別為數值才做計算
            # 文字辨識
            value_img = img[obj['coord'][1]:obj['coord'][3], obj['coord'][0]:obj['coord'][2]]
            value_text = digit_ocr(value_img)

            # 抓出來的文字為數值才做計算
            if(value_text.isdigit()):
                pt1 = [int((obj['coord'][0] + obj['coord'][2]) / 2), int((obj['coord'][1] + obj['coord'][3]) / 2)]
                pt2 = center_coord

                rho = (((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2)) ** 0.5
                angle = int((np.degrees(np.arctan2((pt1[1] - pt2[1]), (pt1[0] - pt2[0]))) + 270) % 360)

                value = int(value_text)
                value_list.append({'value': value, 'angle': angle})

                cv2.putText(result, '{}/{}'.format(value, str(angle)), (obj['coord'][0], obj['coord'][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    # 獲取指針相對應的數值 取最大的兩個數值做計算
    value_list = sorted(value_list, key=itemgetter('value'), reverse=True)
    print(value_list)
    angle_diff = value_list[0]['angle'] - value_list[1]['angle']
    value_diff = value_list[0]['value'] - value_list[1]['value']
    pointer_value = value_list[0]['value'] - (value_list[0]['angle'] - pointer_angle) * (value_diff / angle_diff)

    cv2.putText(result, 'predicted value:{}'.format(str(pointer_value)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    show_img('img', result)
    return result


@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix = False, Path(w).suffix.lower()
    pt, onnx, tflite, pb, graph_def = (suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        if pt:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        elif onnx:
            img = img.astype('float32')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                obj_list = []
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_width=line_thickness)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                        # 獲取座標、類別等資訊
                        x1 = int(xyxy[0].item())
                        y1 = int(xyxy[1].item())
                        x2 = int(xyxy[2].item())
                        y2 = int(xyxy[3].item())

                        # conf_score = conf # 分數
                        obj_class = int(cls) # 類別
                        obj_name = names[int(cls)] # 物件名稱

                        obj_list.append({'class': obj_class, 'name': obj_name, 'coord': [x1, y1, x2, y2]})

                # 數值辨識
                img = cv2.imread(source)
                result = meter_read(img, obj_list)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    cv2.imwrite(save_path[:-4] + '_result.jpg', result)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
