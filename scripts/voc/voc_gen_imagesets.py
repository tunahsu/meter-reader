import os
import random
import glob

# trainval在所有資料中的佔比
trainval_percent = 0.9
# train在trainval的佔比
train_percent = 1

xmlfilepath = '../../data/analog_ammeter_dataset_20211022/Annotations'
txtsavepath = '../../data/analog_ammeter_dataset_20211022/ImageSets/Main'
total_xml = glob.glob(os.path.join(xmlfilepath, '*.xml'))

num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)
ftrainval = open('../../data/analog_ammeter_dataset_20211022/ImageSets/Main/trainval.txt', 'w')
ftest = open('../../data/analog_ammeter_dataset_20211022/ImageSets/Main/test.txt', 'w')
ftrain = open('../../data/analog_ammeter_dataset_20211022/ImageSets/Main/train.txt', 'w')
fval = open('../../data/analog_ammeter_dataset_20211022/ImageSets/Main/val.txt', 'w')

for i in list:
    name=os.path.basename(total_xml[i])[:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
            
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()