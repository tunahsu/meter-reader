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


def get_center(img, result, obj):
    # 擷取物件區域
    img = img[obj['coord'][1]:obj['coord'][3], obj['coord'][0]:obj['coord'][2]]

    if obj['class'] == 1:
        height, width, _ = img.shape
        center_coord = [int(obj['coord'][0] + width * 0.9), int(obj['coord'][1] + height * 0.9)]

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
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(guassian, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 15)
    edges = cv2.Canny(thresh, 100, 150)

    # show_img('Gray', gray)
    # show_img('Blur', blur)
    # show_img('Thresh', thresh)
    # show_img('Edges', edges)

    # 找直線
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, lines=np.array([]), minLineLength=25, maxLineGap=10)
    
    if lines is not None:
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
            print('pointer angle:{}'.format(angle))

            # 畫線
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return angle
    
    print('未偵測到任何指針')
    return None


def meter_read(img, obj_list):  
    for obj in obj_list:
        if obj['class'] in [1, 2, 3, 4]: # 若物件類別為電表
            result = img.copy()

            # 獲取指針、圓
            center_coord = get_center(img, result, obj)
            pointer_angle = get_pointer(img, result, obj, center_coord)

            # 只取一個電表
            break
    if pointer_angle is not None:
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

                    cv2.putText(result, '{}/{}'.format(value, str(angle)), (obj['coord'][0], obj['coord'][1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, cv2.LINE_AA)

        # 獲取指針相對應的數值 取最大的兩個數值做計算
        value_list = sorted(value_list, key=itemgetter('value'), reverse=True)
        print(value_list)
        angle_diff = value_list[0]['angle'] - value_list[-1]['angle']
        value_diff = value_list[0]['value'] - value_list[-1]['value']
        pointer_value = value_list[0]['value'] - (value_list[0]['angle'] - pointer_angle) * (value_diff / angle_diff)

        cv2.putText(result, 'predicted value:{}'.format(str(pointer_value)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, cv2.LINE_AA)
        show_img('img', result)
    return result


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    image_path = FLAGS.image

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

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
    result = meter_read(original_image, obj_list)

    image = utils.draw_bbox(original_image, pred_bbox)
    # image = utils.draw_bbox(image_data*255, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(FLAGS.output, image)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
