# -*- coding: utf-8 -*-
import tensorflow as tf

# 配置神经网络的参数
INPUT_NODE = 784
OUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
# 全连接层的节点个数
FC_SIZE = 512

def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS,
                                        CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(
            "bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        # 使用边长为5，深度32的过滤器，过滤器移动的步长为1，且使用全0填充
        conv1 = tf.nn.conv2d(
            input_tensor, conv1_weights, strides=[1,1,1,1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 实现第二层池化层的前向传播过程
    with tf.name_scope('layer2-pooll'):
        pooll = tf.nn.max_pool(
            relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
    # 声明第三层卷积层的变量并实现前向传播的过程
    # 输出 14*14*64的矩阵
    with tf.name_scope('layer3-pooll'):
            conv2_weights = tf.get_variable(
                "weight", [CONV2_SIZE, CONV2_SIZE, CONV2_DEEP, CONV2_DEEP],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable(
                "bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
            # 使用边长为5，深度为64的过滤器，过滤器移动的步长为1，且使用全0填充
            conv2 = tf.nn.conv2d(
                pooll, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

            # 实现第四层的前向传播过程。这一层和第二层结构一样
    with tf.name_scope('layer4-pool2'):
           pool2 = tf.nn.max_pool(
               relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    # 通过tf.reshape函数将第四层的输出变成一个bantch的向量
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable(
            "weight", [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable(
            "bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    # 声明第六层全连接层的变量并实现前向传播过程
    with tf.variable_scope('layer-fc2'):
        fc2_weights = tf.get_variable(
            "weight", [FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable(
            "biase", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    # 返回第六层的输出
    return logit