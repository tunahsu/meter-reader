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
    text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789').strip()
    # show_img('Thresh', thresh)
    # print(text)
    return(text)


def get_center(img, obj, obj_list):
    # 擷取物件區域
    center_img = img.copy()
    img = img[obj['coord'][1]:obj['coord'][3], obj['coord'][0]:obj['coord'][2]]

    if obj['class'] == 7:
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

    elif obj['class'] == 8:
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


def get_polar_img(img, obj, center_coord):
    height, width, _ = img.shape
    dist = center_coord[1] - obj['coord'][1]

    # 圖片逆時針90度
    img_rotate = cv2.transpose(img)
    img_rotate = cv2.flip(img_rotate, 0)

    # 圓心座標逆時針90度
    center_coord_rotate = [center_coord[1], width - center_coord[0]]

    # 極座標轉換 空的部分補上白色
    polar_img = cv2.linearPolar(img_rotate, (center_coord_rotate[0], center_coord_rotate[1]), dist, cv2.INTER_LINEAR)
    # polar_img[np.where((polar_img == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    # 方形電表取其90~180度
    if obj['class'] == 7:
        top = polar_img.shape[0] // 4
        bottom = polar_img.shape[0] // 2
        polar_img = polar_img[top:bottom,]

    return polar_img


def get_pointer(obj, polar_img):
    polar_gray = cv2.cvtColor(polar_img, cv2.COLOR_BGR2GRAY)
    ret, polar_binary = cv2.threshold(polar_gray, 150, 255, cv2.THRESH_BINARY)

    # 透過累加器取出黑色pixel最多的列
    acc_max = 0
    for i in range(polar_binary.shape[0]):
        acc = 0
        for j in range(polar_binary.shape[1]):
            if polar_binary[i][j] == 0:
                acc += 1
        if acc > acc_max:
            acc_max = acc
            pointer_pos = i

    # 不同的電表有不同的角度計算範圍
    if obj['class'] == 7:
        pointer_angle = (pointer_pos / polar_img.shape[0]) * 90 + 90
    elif obj['class'] == 8:
        pointer_angle = (pointer_pos / polar_img.shape[0]) * 360

    cv2.line(polar_img, (0, pointer_pos), (polar_img.shape[1], pointer_pos), (255, 0, 0), 2)
    show_img('Pointer', polar_img)

    return pointer_angle, polar_img

def get_values(img, obj_list, center_coord, pointer_angle):
    value_list = []
    for obj in obj_list:
        # 若物件類別為數值才做計算
        if obj['class'] == 0:
            value_img = img[obj['coord'][1]:obj['coord'][3], obj['coord'][0]:obj['coord'][2]]
            value_text = digit_ocr(value_img)

            # 文字內容為數字才做計算
            if(value_text.isdigit()):
                pts = [
                    [obj['coord'][0], obj['coord'][1]],
                    [obj['coord'][0], obj['coord'][3]],
                    [obj['coord'][2], obj['coord'][1]],
                    [obj['coord'][2], obj['coord'][3]]
                ]
                
                # 計算數值物件四個角與圓心的角度
                angles = []
                for i in range(4):
                    rho = (((pts[i][0] - center_coord[0]) ** 2) + ((pts[i][1] - center_coord[1]) ** 2)) ** 0.5
                    angle = (np.degrees(np.arctan2((pts[i][1] - center_coord[1]), (pts[i][0] - center_coord[0]))) + 270) % 360
                    angles.append(angle)
                angle_max = max(angles)
                angle_min = min(angles)

                # 若指針角度落在其之間則忽略此數值
                if pointer_angle < angle_max and pointer_angle > angle_min:
                    continue

                # 計算數值物件中心與圓心的角度
                value_coord = [(obj['coord'][0] + obj['coord'][2]) // 2, (obj['coord'][1] + obj['coord'][3]) // 2]
                rho = (((value_coord[0] - center_coord[0]) ** 2) + ((value_coord[1] - center_coord[1]) ** 2)) ** 0.5
                angle = (np.degrees(np.arctan2((value_coord[1] - center_coord[1]), (value_coord[0] - center_coord[0]))) + 270) % 360

                value = float(value_text)
                value_list.append([value, angle])

    return(value_list)

def meter_read(img, obj_list):  
    for obj in obj_list:
        if obj['class'] in [7, 8]:
            # 獲取指針、圓若物件類別為電表
            center_img, center_coord = get_center(img, obj, obj_list)
            polar_img = get_polar_img(img, obj, center_coord)
            pointer_angle, polar_img = get_pointer(obj, polar_img)
            value_list = get_values(img, obj_list, center_coord, pointer_angle)

            # 將指針加入value_list中
            value0 = [None, pointer_angle]
            value_list.append(value0)
            value_list = sorted(value_list, key=itemgetter(1))

            # 取離指針最近得兩個刻度作為角度法的參考值
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

            # 角度法計算
            scale_per_degree = (value2[0] - value1[0]) / (value2[1] - value1[1])
            value0 = [value2[0] + (pointer_angle - value2[1]) * scale_per_degree, pointer_angle]
            value0[0] = value0[0] if value0[0] > 0 else 0
            
            print(value1, value2, value0)     

            return value0, img, center_img, polar_img
    return None 


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
    
    try:
        value0, img, center_img, polar_img = meter_read(original_image, obj_list)

        plt.subplot(1, 3, 1)
        plt.title('center')
        plt.axis('off')
        plt.imshow(center_img)
        plt.subplot(1, 3, 2)
        plt.title('angle: {}'.format(str(round(value0[1], 2))))
        plt.axis('off')
        plt.imshow(polar_img)
        plt.subplot(1, 3, 3)
        plt.title('value: {}'.format(str(round(value0[0], 2))))
        plt.axis('off')
        plt.imshow(img)
        plt.savefig('result.png', dpi=300)
        plt.show()
    except Exception as e:
        print("辨識失敗: {}".format(e))
        # print([obj['class'] for obj in obj_list])

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
