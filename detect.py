import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# my imports
from operator import itemgetter
import matplotlib.pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './data/kite.jpg', 'path to input image')
flags.DEFINE_string('output', 'result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')


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


def get_center(img, obj, obj_list):
    # 擷取物件區域
    center_img = img.copy()
    img = img[obj['coord'][1]:obj['coord'][3], obj['coord'][0]:obj['coord'][2]]

    if obj['class'] in [1, 2]:
        # 任取圓上三點計算圓心
        value_list = [value for value in obj_list if value['class'] == 0]
        pt1 = [(value_list[0]['coord'][0] + value_list[0]['coord'][2]) / 2, (value_list[0]['coord'][1] + value_list[0]['coord'][3]) / 2]
        pt2 = [(value_list[1]['coord'][0] + value_list[1]['coord'][2]) / 2, (value_list[1]['coord'][1] + value_list[1]['coord'][3]) / 2]
        pt3 = [(value_list[2]['coord'][0] + value_list[2]['coord'][2]) / 2, (value_list[2]['coord'][1] + value_list[2]['coord'][3]) / 2]

        v21 = [pt2[0] - pt1[0], pt2[1] - pt1[1]]
        pt21 = [(pt2[0] + pt1[0]) / 2, (pt2[1] + pt1[1]) / 2]
        c1 = v21[0] * pt21[0] + v21[1] * pt21[1]

        v32 = [pt3[0] - pt2[0], pt3[1] - pt2[1]]
        pt32 = [(pt3[0] + pt2[0]) / 2, (pt3[1] + pt2[1]) / 2]
        c2 = v32[0] * pt32[0] + v32[1] * pt32[1]            

        A = np.array([[v21[0], v21[1]], [v32[0], v32[1]]])
        B = np.array([c1, c2])
        A_inv = np.linalg.inv(A)
        ans = A_inv.dot(B)

        # 取圓心座標、半徑
        center_coord = [int(ans[0]), int(ans[1])]
        center_radius = int(((pt1[0] - center_coord[0]) ** 2 + (pt1[1] - center_coord[1])  ** 2) ** 0.5)

    elif obj['class'] == 3:
        # 霍夫找圓
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=200, minRadius=100, maxRadius=600)
        circles = np.uint16(np.around(circles))
        circle = circles[0][0]

        # 取圓心座標、半徑 含座標校正
        center_coord = [int(obj['coord'][0] + circle[0]), int(obj['coord'][1] + circle[1])]
        center_radius = circle[2]

    # 畫圓
    cv2.circle(center_img,(center_coord[0], center_coord[1]), center_radius, (255, 0, 0), 2)
    cv2.circle(center_img, (center_coord[0], center_coord[1]), 2, (255, 0, 0), 2)
    show_img('Center', center_img)

    return center_img, center_coord


def get_pointer_values(meter, img, polar_img, obj_list, center_coord):
    value_list = []
    value_xlist = []
    value_ylist = []

    for obj in obj_list:
        if obj['class'] == 0: # 若物件類別為數值才做計算
            # 文字辨識
            value_img = img[obj['coord'][1]:obj['coord'][3], obj['coord'][0]:obj['coord'][2]]
            value_text = digit_ocr(value_img)

            # 文字內容為數字才做計算
            if(value_text.isdigit()):
                # 將數值物件左上/下、右上/下座標通通抓出來
                value_top_left = get_polar_coord(img, center_coord, [obj['coord'][0], obj['coord'][1]])
                value_top_right = get_polar_coord(img, center_coord, [obj['coord'][2], obj['coord'][1]])
                value_bottom_left = get_polar_coord(img, center_coord, [obj['coord'][0], obj['coord'][3]])
                value_bottom_right = get_polar_coord(img, center_coord, [obj['coord'][2], obj['coord'][3]])
                value_xlist.extend([value_top_left[1], value_top_right[1], value_bottom_left[1], value_bottom_right[1]])

                # 獲取數值物件在polar_image中的"最"左/上/右/下位置
                sec_roi_min_x = min(value_top_left[1], value_top_right[1], value_bottom_left[1], value_bottom_right[1])
                sec_roi_max_x = max(value_top_left[1], value_top_right[1], value_bottom_left[1], value_bottom_right[1])
                sec_roi_min_y = min(value_top_left[0], value_top_right[0], value_bottom_left[0], value_bottom_right[0])
                sec_roi_max_y = max(value_top_left[0], value_top_right[0], value_bottom_left[0], value_bottom_right[0])
                
                # 截出感興趣的區域
                sec_roi = polar_img[sec_roi_min_y:sec_roi_max_y, sec_roi_min_x:(2 * sec_roi_max_x - sec_roi_min_x)]
                sec_roi_gray = cv2.cvtColor(sec_roi, cv2.COLOR_BGR2GRAY)
                ret, sec_roi_thresh = cv2.threshold(sec_roi_gray, 150, 255, cv2.THRESH_BINARY)
                # show_img('Second ROI', sec_roi_thresh)

                # 透過累加器取出黑色pixel最多的列   
                max_acc = 0
                for i in range(sec_roi_thresh.shape[0]):
                    acc = 0
                    for j in range(sec_roi_thresh.shape[1]):
                        if sec_roi_thresh[i][j] == 0:
                            acc += 1
                    if acc > max_acc:
                        max_acc = acc
                        value_pos = i

                value_pos += sec_roi_min_y
                value_list.append([int(value_text), value_pos])

    min_x = min(value_xlist)
    max_x = max(value_xlist)

    if meter['class'] in [1, 2]:
        # 方形電表避免截到黑色區域 進一步限制polar_image的長
        min_y = get_polar_coord(img, center_coord, [center_coord[0] - 5, center_coord[1] + 1])[0]
        max_y = get_polar_coord(img, center_coord, [center_coord[0] + 1, center_coord[1] - 5])[0]

    elif meter['class'] == 3:
        # 圓形電表抓原本的長就好
        min_y = 0
        max_y = polar_img.shape[0]

    # 截出感興趣的區域
    polar_crop = polar_img[min_y:max_y, min_x:max_x + (max_x - min_x)]
    polar_gray = cv2.cvtColor(polar_crop, cv2.COLOR_BGR2GRAY)
    ret, polar_thresh = cv2.threshold(polar_gray, 150, 255, cv2.THRESH_BINARY)
    
    # 透過累加器取出黑色pixel最多的列
    max_acc = 0
    for i in range(polar_thresh.shape[0]):
        acc = 0
        for j in range(polar_thresh.shape[1]):
            if polar_thresh[i][j] == 0:
                acc += 1
        if acc > max_acc:
            max_acc = acc
            pointer_pos = i

    cv2.line(polar_crop, (0, pointer_pos), (polar_crop.shape[1], pointer_pos), (255, 0, 0), 2)

    for i in range(len(value_list)):
        # 根據感興趣的區塊對數值物件的Y軸位置做校正
        value_list[i][1] -= min_y
        cv2.line(polar_crop, (0, value_list[i][1]), (polar_crop.shape[1], value_list[i][1]), (0, 255, 0), 2)
        cv2.putText(polar_crop, '{}'.format(str(value_list[i][0])), (0, value_list[i][1]), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 1, cv2.LINE_AA)

    return polar_crop, pointer_pos, value_list


def get_polar_img(img, center_coord):
    height, width, _ = img.shape

    # 圖片逆時針90度
    img_rotate = cv2.transpose(img)
    img_rotate = cv2.flip(img_rotate, 0)

    # 圓心座標逆時針90度
    center_coord_rotate = [center_coord[1], width - center_coord[0]]

    polar_img = cv2.linearPolar(img_rotate, (center_coord_rotate[0], center_coord_rotate[1]), max(width, height), cv2.INTER_LINEAR)
    return polar_img


def get_polar_coord(img, center_coord, coord):
    height, width, _ = img.shape

    # 座標逆時針90度
    coord_rotate = [coord[1], width - coord[0]]

    # 圓心座標逆時針90度
    center_coord_rotate = [center_coord[1], width - center_coord[0]]

    # 計算轉換後的坐標
    rho = (((coord_rotate[0] - center_coord_rotate[0]) ** 2) + ((coord_rotate[1] - center_coord_rotate[1]) ** 2)) ** 0.5
    angle = int((np.degrees(np.arctan2((coord_rotate[1] - center_coord_rotate[1]), (coord_rotate[0] - center_coord_rotate[0]))) + 360) % 360)
    polar_coord = [int(height * (angle / 360)), int(rho)]
    return polar_coord


def meter_read(img, obj_list):  
    for obj in obj_list:
        if obj['class'] in [1, 2, 3, 4]:
            # 獲取指針、圓若物件類別為電表
            meter = obj
            center_img, center_coord = get_center(img, obj, obj_list)
            polar_img = get_polar_img(img, center_coord)
            polar_crop, pointer_pos, value_list = get_pointer_values(meter, img, polar_img, obj_list, center_coord)

            # 指針跟刻度位置一樣的就刪掉
            for i in range(len(value_list)):
                if value_list[i][1] == pointer_pos:
                    del value_list[i]
                    break

            # 將指針加入value_list中
            value0 = [None, pointer_pos]
            value_list.append(value0)
            value_list = sorted(value_list, key=itemgetter(1))
            print(value_list)

            for i in range(len(value_list)):
                if value_list[i][1] == value0[1]:
                    if i == 0:
                        value1 = value_list[i + 1]
                        value2 = value_list[i + 2]
                    elif i == len(value_list) - 1:
                        value1 = value_list[i - 1]
                        value2 = value_list[i - 2]
                    else:
                        value1 = value_list[i - 1]
                        value2 = value_list[i + 1]

            scale_per_pixel = (value2[0] - value1[0]) / (value2[1] - value1[1])
            value0 = [value2[0] + (pointer_pos - value2[1]) * scale_per_pixel, pointer_pos]
            value0[0] = value0[0] if value0[0] > 0 else 0
            
            print(value1, value2, value0)
            cv2.putText(polar_crop, '{}'.format(str(round(value0[0], 2))), (0, value0[1]), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 0), 1, cv2.LINE_AA)
            show_img('Polar crop', polar_crop)
            # 只取一個電表
            break  

    return center_img, polar_crop


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    image_path = FLAGS.image

    original_image = cv2.imread(image_path)
    # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    # image_data = image_data[np.newaxis, ...].astype(np.float32)

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
        interpreter.set_tensor(input_details[0]['index'], images_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
            boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
        else:
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    
    # 將所有物件送進 meter_read() 做數值自動判讀
    obj_list = []
    for i in range(valid_detections.numpy()[0]):
        box = boxes.numpy()[0][i]
        y1 = int(box[0] * original_image.shape[0])
        x1 = int(box[1] * original_image.shape[1])
        y2 = int(box[2] * original_image.shape[0])
        x2 = int(box[3] * original_image.shape[1])
        obj_class = int(classes.numpy()[0][i])
        obj_list.append({'class': obj_class, 'coord': [x1, y1, x2, y2]})
    center_img, polar_crop = meter_read(original_image, obj_list)

    image = utils.draw_bbox(original_image, pred_bbox)
    # image = utils.draw_bbox(image_data*255, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(center_img)
    plt.subplot(1, 3, 3)
    plt.imshow(polar_crop)
    plt.savefig('result.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
