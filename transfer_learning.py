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
def get_imge_path(imgae_lists, image_dir, label_name, index, category):
    label_lists = imgae_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)

    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path

def get_bottleneck_path(imge_lists, label_name, index, category):
    return get_imge_path(imge_lists, CACHE_DIR,label_name, index, category)\
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
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(
        image_lists, label_name, index, category)
    # 如果这个特征向量文件不存在，则通过Inception-v3模型来计算特征向量，并将计算的结果存入文件。
    if not os.path.exists(bottleneck_path):
        image_path = get_imge_path(
            image_lists, INPUT_DATA, label_name, index, category)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        # 通过Inception-v3模型计算特征向量
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottlenck_file:
            bottlenck_file.write(bottleck_string)
    else:
        # 直接从文件中获取图片相应的特征向量
        with open(bottleneck_path, 'r') as bottlenck_file:
            bottleck_string = bottlenck_file.read()
        bottleneck_values = [float(x) for x in bottleck_string.split(',')]
    return bottleneck_values
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category,
                                  jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor
        )
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

# 获取全部数据
def get_test_bottlenecks(sess, image_lists, n_classes,jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(
            image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(
               sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor
            )
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

def main():
    image_lists = creat_imge_lists(TEST_PERCENTAGE,VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFormString(f.read())
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def,
        return_elements=[BOTTLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME]
    )
    bottleneck_input = tf.placeholder(
        tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(
        tf.float32, [None, n_classes], name='GroundTruthInput')
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal(
            [BOTTLENECK_TENSOR_SIZE, n_classes], stddev = 0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)
    # 定义交叉损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits, ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.arg_max(final_tensor, 1), tf.arg_max(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        # 训练过程
        for i in range(STEPS):
            train_bottlenecks, train_ground_truth = \
            get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor
            )
            sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
            if i % 100 == 0 or i+1 ==STEPS:
                validation_bottlenecks, validation_ground_truth = \
                get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor
                )
                validation_accuracy = sess.run(
                    evaluation_step, feed_dict={
                        bottleneck_input: validation_bottlenecks,
                        ground_truth_input: validation_ground_truth}
                )
                print('Step %d: Validation accuracy on random sampled' '%d examples = %.1f%%'
                      % (i, BATCH, validation_accuracy * 100))
            # 在最后的测试数据上测试正确率
            test_bottlenecks, test_ground_truth = get_test_bottlenecks(
                sess, image_lists, n_classes,jpeg_data_tensor,bottleneck_tensor)
            test_accuracy = sess.run(evaluation_step,
                                     feed_dict={bottleneck_input:test_bottlenecks,
                                     ground_truth_input: test_ground_truth})
            print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

if __name__ == '__main__':
    tf.app.run()



