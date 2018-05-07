# -*- coding: utf-8 -*_

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# Inception-v3模型瓶颈层的节点数
BOTTLENECK_TENSOR_SIZE = 2048

BOTTLENECK_TENSOR_NAME = 'pool_3/reshape:0'

JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

MODEL_DIR = '/path/to/model'
MODEL_FILE = 'classify_image_graph_def.pb'
CACHE_DIR = '/tmp/bottleneck'

INPUT_DATA = '/path/to/flower_data'
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100
def creat_imge_lists(testing_percentage, validation_percentage):
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in  sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'jpg', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extensions in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.'+extensions)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        label_name = dir_name.lower()
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance <(testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result
def get_img_path(imgae_lists, image_dir, label_name, index, category):
    label_lists = imgae_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)

    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path

def get_bottleneck_path(imge_lists, label_name, index, category):
    return get_img_path(imge_lists, CACHE_DIR,label_name, index, category)\
       + '.txt'
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor,{image_data_tensor:image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return  bottleneck_values

def get_or_create_bottleneck(sess, image_lists, label_name, index,category,
                             jpeg_data_tensor, bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(
        image_lists, label_name, index, category)



