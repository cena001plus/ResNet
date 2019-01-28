​
import tensorflow as tf


def conv_op(x, name, n_out, training, useBN, kh=3, kw=3, dh=1, dw=1, padding="SAME", activation=tf.nn.relu):
    '''
    x:输入
    kh,kw:卷集核的大小
    n_out:输出的通道数
    dh,dw:strides大小
    name:op的名字

    '''
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        w = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b = tf.get_variable(scope + "b", shape=[n_out], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.01))
        conv = tf.nn.conv2d(x, w, [1, dh, dw, 1], padding=padding)
        z = tf.nn.bias_add(conv, b)
        if useBN:
            z = tf.layers.batch_normalization(z, trainable=training)
        if activation:
            z = activation(z)
        return z


def max_pool_op(x, name, kh=2, kw=2, dh=2, dw=2, padding="SAME"):
    return tf.nn.max_pool(x,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding=padding,
                          name=name)


def avg_pool_op(x, name, kh=2, kw=2, dh=2, dw=2, padding="SAME"):
    return tf.nn.avg_pool(x,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding=padding,
                          name=name)


def fc_op(x, name, n_out, activation=tf.nn.relu):
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        w = tf.get_variable(scope + "w", shape=[n_in, n_out],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(scope + "b", shape=[n_out], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.01))

        fc = tf.matmul(x, w) + b

        out = activation(fc)

    return fc, out



def res_block_layers(x, name, n_out_list, change_dimension=False, block_stride=1):
    if change_dimension:
        short_cut_conv = conv_op(x, name + "_ShortcutConv", n_out_list[1], training=True, useBN=True, kh=1, kw=1,
                                 dh=block_stride, dw=block_stride,
                                 padding="SAME", activation=None)
    else:
        short_cut_conv = x

    block_conv_1 = conv_op(x, name + "_lovalConv1", n_out_list[0], training=True, useBN=True, kh=1, kw=1,
                           dh=block_stride, dw=block_stride,
                           padding="SAME", activation=tf.nn.relu)

    block_conv_2 = conv_op(block_conv_1, name + "_lovalConv2", n_out_list[0], training=True, useBN=True, kh=3, kw=3,
                           dh=1, dw=1,
                           padding="SAME", activation=tf.nn.relu)

    block_conv_3 = conv_op(block_conv_2, name + "_lovalConv3", n_out_list[1], training=True, useBN=True, kh=1, kw=1,
                           dh=1, dw=1,
                           padding="SAME", activation=None)

    block_res = tf.add(short_cut_conv, block_conv_3)
    res = tf.nn.relu(block_res)
    return res


def inference(x, batch_size, n_classes, training=True, usBN=True):
    '''
    build model，ResNet50
    args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    return:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    conv1 = conv_op(x, "conv1", 64, training, usBN, 3, 3, 2, 2)
    pool1 = max_pool_op(conv1, "pool1", kh=3, kw=3)

    block1_1 = res_block_layers(pool1, "block1_1", [64, 256], True, 1)
    block1_2 = res_block_layers(block1_1, "block1_2", [64, 256], False, 1)
    block1_3 = res_block_layers(block1_2, "block1_3", [64, 256], False, 1)

    block2_1 = res_block_layers(block1_3, "block2_1", [128, 512], True, 2)
    block2_2 = res_block_layers(block2_1, "block2_2", [128, 512], False, 1)
    block2_3 = res_block_layers(block2_2, "block2_3", [128, 512], False, 1)
    block2_4 = res_block_layers(block2_3, "block2_4", [128, 512], False, 1)

    block3_1 = res_block_layers(block2_4, "block3_1", [256, 1024], True, 2)
    block3_2 = res_block_layers(block3_1, "block3_2", [256, 1024], False, 1)
    block3_3 = res_block_layers(block3_2, "block3_3", [256, 1024], False, 1)
    block3_4 = res_block_layers(block3_3, "block3_4", [256, 1024], False, 1)
    block3_5 = res_block_layers(block3_4, "block3_5", [256, 1024], False, 1)
    block3_6 = res_block_layers(block3_5, "block3_6", [256, 1024], False, 1)

    block4_1 = res_block_layers(block3_6, "block4_1", [512, 2048], True, 2)
    block4_2 = res_block_layers(block4_1, "block4_2", [512, 2048], False, 1)
    block4_3 = res_block_layers(block4_2, "block4_3", [512, 2048], False, 1)

    pool2 = avg_pool_op(block4_3, "pool2", kh=7, kw=7, dh=1, dw=1, padding="SAME")
    shape = pool2.get_shape()
    fc_in = tf.reshape(pool2, [-1, shape[1].value * shape[2].value * shape[3].value])
    logits, prob = fc_op(fc_in, "fc1", n_classes, activation=tf.nn.softmax)

    #return logits, prob
    return logits

#-------------------------------------------------------------------------
def losses(logits,labels):
    # loss
    # args： logits，net's output ; labels，the real value，0 or 1
    # return：loss，the error between prediction and the real value
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels = labels, name='xentropy_re_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss


#-------------------------------------------------------------------------
def trainning(loss, learning_rate):
    # trainning
    # args：logits，net's output; labels，the real value，0 or 1
    # return：
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


#---------------------------------------------------------------------------
def evaluation(logits, labels):
    # acc calc
    # args：logits，net's output; labels，the real value，0 or 1
    # return：accuracy, The average accuracy of the current step, that is, how many
    # images in these batches are correctly classified.
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy

​