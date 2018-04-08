import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import sys

sys.path.append('./Network')
from vgg16 import VGG16
from datagenerator import ImageDataGenerator

"""
Configuration settings
"""

# Path to the textfiles for the trainings and validation set
train_file = '/path/to/train.txt'
val_file = '/path/to/val.txt'

# Learning params
# learning_rate = 0.01
learning_rate = 1./255
num_epochs = 100
batch_size = 32

# Network params
dropout_rate = 0.5
num_classes = 2
train_layers = ['fc8', 'fc7']

# How often we want to write the tf.summary data to disk
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "./summary/vgg16"
checkpoint_path = "./checkpoint/vgg16"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)
if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = VGG16(x, keep_prob, num_classes, train_layers, './Network/pretrained_model/vgg16.npy', name='VGG16')

# Link variable to model output
score = model.fc_final

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[1] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Train_finetune op
with tf.name_scope("train_finetune"):
    optimizer_finetune = tf.train.MomentumOptimizer(learning_rate=1e-4, momentum=0.9)
    train_finetune_op = optimizer_finetune.minimize(loss)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Initalize the data generator seperately for the training and validation set
    train_generator = ImageDataGenerator(
        csv_path='../../Species/data/train_labels.csv',
        file_path='../../Species/data/train/',
        image_format='jpg',
        resize_factor=[224, 224],
        num_classes=num_classes,
        num_epochs=None,
        session=sess,
        batch_size=batch_size,
        shuffle=True,
        aug=True,
        name_scope='train')

    val_generator = ImageDataGenerator(
        csv_path='../../Species/data/validation_labels.csv',
        file_path='../../Species/data/validation/',
        image_format='jpg',
        resize_factor=[224, 224],
        num_classes=num_classes,
        num_epochs=None,
        session=sess,
        batch_size=batch_size,
        shuffle=False,
        aug=False,
        name_scope='validation')

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Thread 관리
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        # Loop over number of epochs
        for epoch in range(num_epochs):

            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
            step = 1

            while step < train_batches_per_epoch:

                # Get a batch of images and labels
                batch_xs, batch_ys = train_generator.next_batch()

                # And run the training op
                sess.run(train_op, feed_dict={x: batch_xs,
                                              y: batch_ys,
                                              keep_prob: dropout_rate})

                # Generate summary with the current batch of data and write to file
                if step % display_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: batch_xs,
                                                            y: batch_ys,
                                                            keep_prob: 1.})
                    writer.add_summary(s, epoch * train_batches_per_epoch + step)

                step += 1

            if (epoch + 1) % 10 == 0:
                # Validate the model on the entire validation set
                print("{} Start validation".format(datetime.now()))
                test_acc = 0.
                test_count = 0
                for _ in range(val_batches_per_epoch):
                    batch_tx, batch_ty = val_generator.next_batch()
                    acc = sess.run(accuracy, feed_dict={x: batch_tx,
                                                        y: batch_ty,
                                                        keep_prob: 1.})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))
                print("{} Saving checkpoint of model...".format(datetime.now()))

                # save checkpoint of the model
                checkpoint_name = os.path.join(checkpoint_path, 'vgg16_model_epoch' + str(epoch + 1) + '.ckpt')
                save_path = saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

    except tf.errors.OutOfRangeError:
        print('Saving')
        checkpoint_name = os.path.join(checkpoint_path, 'vgg16_model_epoch' + str(epoch + 1) + '.ckpt')
        saver.save(sess, checkpoint_name)
        print('Done training for %s epochs, %s steps.' % (str(epoch + 1), "Final"))

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    coord.request_stop()
    coord.join(threads)