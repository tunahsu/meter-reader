import sys
import os

from absl import app, flags
from absl.flags import FLAGS
from lxml import etree


flags.DEFINE_string('image_dir', '../../data/analog_ammeter_dataset_20211022/JPEGImages', 'path to image dir')
flags.DEFINE_string('anno_dir', '../../data/analog_ammeter_dataset_20211022/Annotations', 'path to anno dir')
flags.DEFINE_string('train_list_txt', '../../data/analog_ammeter_dataset_20211022/ImageSets/Main/train.txt', 'path to a set of train')
flags.DEFINE_string('val_list_txt', '../../data/analog_ammeter_dataset_20211022/ImageSets/Main/test.txt', 'path to a set of val')
flags.DEFINE_string('classes', '../../data/classes/analog_ammeter.names', 'path to a list of class names')
flags.DEFINE_string('train_output', '../../data/dataset/analog_ammeter_train.txt', 'path to a file for train')
flags.DEFINE_string('val_output', '../../data/dataset/analog_ammeter_val.txt', 'path to a file for val')

flags.DEFINE_boolean('no_val', False, 'if uses this flag, it does not convert a list of val')


def convert_annotation(list_txt, output_path, image_dir, anno_dir, class_names):
    IMAGE_EXT = '.jpg'
    ANNO_EXT = '.xml'

    with open(list_txt, 'r') as f, open(output_path, 'w') as wf:
        while True:
            line = f.readline().strip()
            if line is None or not line:
                break
            im_p = os.path.join(image_dir, line + IMAGE_EXT)
            an_p = os.path.join(anno_dir, line + ANNO_EXT)

            # Get annotation.
            root = etree.parse(an_p).getroot()
            bboxes = root.xpath('//object/bndbox')
            names = root.xpath('//object/name')

            box_annotations = []
            for b, n in zip(bboxes, names):
                name = n.text
                class_idx = class_names.index(name)

                xmin = b.find('xmin').text
                ymin = b.find('ymin').text
                xmax = b.find('xmax').text
                ymax = b.find('ymax').text
                box_annotations.append(','.join([str(xmin), str(ymin), str(xmax), str(ymax), str(class_idx)]))
            
            annotation = os.path.abspath(im_p) + ' ' + ' '.join(box_annotations) + '\n'

            wf.write(annotation)


def convert_voc(image_dir, anno_dir, train_list_txt, val_list_txt, classes, train_output, val_output, no_val):
    IMAGE_EXT = '.jpg'
    ANNO_EXT = '.xml'

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]

    # Training set.
    convert_annotation(train_list_txt, train_output, image_dir, anno_dir, class_names)

    if no_val:
        return

    # Validation set.
    convert_annotation(val_list_txt, val_output, image_dir, anno_dir, class_names)


def main(_argv):
    convert_voc(FLAGS.image_dir, FLAGS.anno_dir, FLAGS.train_list_txt, FLAGS.val_list_txt, FLAGS.classes, FLAGS.train_output, FLAGS.val_output, FLAGS.no_val)
    print("Complete convert voc data!")


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass    
