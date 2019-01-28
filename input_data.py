​
import tensorflow as tf
import numpy as np
import os

train_dir = 'D:/XXXXX/XXXXXX/train/'


def get_files(file_dir):
    '''
    args:
        file_dir : file directory
    return:
        list of images and labels
    '''
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split('.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)

    print('There are %d cats\n There are %d dogs' % (len(cats), len(dogs)))

    # hstack => axis = 1,eg: cats = [[123234]] dogs = [[321432]]
    # => [[123234321432]]
    # label_cats = [1,2,3]  label_dogs = [3,2,1]
    # => [1,2,3,3,2,1]
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    # [
    # [aaa,bbb,ccc],
    # [0,0,1]
    #        ]

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    # [
    # [aaa,0],
    # [bbb,0],
    # [ccc,1],
    # ]
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    args:
        images:list type
        label:list type
        image_W:image width
        image_H:image height
        batch_size:batch_size
        capacity:the maximum elements in queue
    return:
        image_batch: 4D tensor [batch_size , width , height , 3], dtype = tf.float32
        label_batch: 1D tensor [batch_size], dtype = tf.int32
    '''
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    # [
    # [aaa,bbb,ccc],
    # [0,1,0]
    # ]
    print("image,label shape : ", image.get_shape(), label.get_shape())
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])

    # step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch

# if __name__ == '__main__':
# train, train_label = get_files(train_dir)
# get_batch(train, train_label, 208,208,16,2000)

​