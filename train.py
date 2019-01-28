​
import os
import numpy as np
import tensorflow as tf
import input_data
import model_resnet50

# super parameter
N_CLASSES = 2
# resize the image, if the input image is too large, training will be very slow.
IMG_W = 224
IMG_H = 224
BATCH_SIZE = 32
CAPACITY = 200
# with current parameters, it is suggested to use MAX_STEP>10k
MAX_STEP = 20
# with current parameters, it is suggested to use learning rate<0.0001
learning_rate = 0.0001


# Trainning
def run_training():
    # you need to change the directories to yours.
    train_dir = 'D:/tensorflow/practicePlus/ResNet/train/'
    # val_dir = 'D:/tensorflow/practicePlus/cats_vs_dogs/test'
    logs_train_dir = 'D:/tensorflow/practicePlus/ResNet/save/'

    train, train_label = input_data.get_files(train_dir)
    # val, val_label = input_data.get_files(val_dir)
    train_batch, train_label_batch = input_data.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    # val_batch, val_label_batch = input_data.get_batch(val, val_label,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)

    # train
    train_logits = model_resnet50.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model_resnet50.losses(train_logits, train_label_batch)
    train_op = model_resnet50.trainning(train_loss, learning_rate)
    train_acc = model_resnet50.evaluation(train_logits, train_label_batch)

    # validation
    # test_logits = model.inference(val_batch,BATCH_SIZE,N_CLASSES)
    # test_loss = model.losses(test_logits, val_label_batch)
    # test_acc = model.evaluation(test_logits, val_label_batch)

    summary_op = tf.summary.merge_all()

    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # batch trainning
    try:
        # one step one batch
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

            # print loss and acc each 10 step, record log and write at same time
            if step % 10 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            # save modle each 500 steps
            if ((step == 500) or ((step + 1) == MAX_STEP)):
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    finally:
        coord.request_stop()


# --------------------------------------
if __name__ == '__main__':
    run_training()

​