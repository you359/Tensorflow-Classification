import tensorflow as tf
import numpy as np
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-25, 25), # rotate by -25 to +25 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 1.0
                    iaa.AverageBlur(k=(2, 5)), # blur image using local means with kernel sizes between 2 and 5
                    iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 3 and 5
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges
                #sometimes(iaa.OneOf([
                #    iaa.EdgeDetect(alpha=(0, 0.7)),
                #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                #])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                #iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 0.2)),
                sometimes(iaa.ElasticTransformation(alpha=(0.1, 1.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03))) # sometimes move parts of the image around
            ],
            random_order=True
        )
    ],
    random_order=True
)

class ImageDataGenerator:
    def __init__(self, csv_path, file_path, image_format, resize_factor, num_classes, num_epochs, session, batch_size, shuffle=False, aug=False, name_scope='train'):
        # Init params
        self.file_path = file_path              # 이미지 파일 path
        self.image_format = image_format        # 이미지 파일 format
        self.sess = session                     # TF session
        self.shuffle = shuffle                  # Shuffling 여부
        self.num_epochs = num_epochs            # epoch 개수
        self.name_scope = name_scope            # name scope tensor 이름 정의
        self.batch_size = batch_size            # batch_size
        self.aug = aug                          # augmentation 수행 여부
        self.resize_factor = resize_factor      # 이미지 resize 크기
        self.num_classes = num_classes          # 클래스 개수
        self.filenames = []
        self.labels = []

        self.read_labeled_image_list_from_csv(csv_path)
        self.input_data = self.input_pipeline()
        self.aug_flag = True

    # csv file Reader
    def read_labeled_image_list_from_csv(self, csv_path):
        """Reads a .csv file containing pathes and labeles
        Args:
           image_list_file: a .txt file with one /path/to/image per line
           label: optionally, if set label will be pasted after each line
        Returns:
           List with all filenames in file image_list_file
        """
        f = open(csv_path, 'r')
        except_title = False

        for line in f:
            # print(line)
            if except_title:
                self.filename, self.label = line[:-1].split(',')
                self.filename = self.file_path + self.filename + '.' + self.image_format

                self.filenames.append(self.filename)
                self.labels.append(int(self.label))

            if not except_title:
                except_title = True
        self.data_size = len(self.labels)

    def read_images_from_disk(self, input_queue):
        """Consumes a single filename and label as a ' '-delimited string.
        Args:
          filename_and_label_tensor: A scalar string tensor.
        Returns:
          Two tensors: the decoded image, and the string label.
        """
        with tf.name_scope('decode') as scope:
            label = input_queue[1]
            file_contents = tf.read_file(input_queue[0])
            example = tf.image.decode_jpeg(file_contents, channels=3)
            example = tf.image.resize_images(
                example,
                size=[self.resize_factor[0], self.resize_factor[1]],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return example, label

    def input_pipeline(self):
        images = tf.convert_to_tensor(self.filenames, dtype=tf.string)
        labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)

        with tf.name_scope(self.name_scope + '-InputPipeline') as scope:
            # Makes an input queue
            input_queue = tf.train.slice_input_producer(
                [images, labels],
                num_epochs=self.num_epochs,
                shuffle=self.shuffle,
                name='filename_queue')

            image, label = self.read_images_from_disk(input_queue)

            min_after_dequeue = 10000
            capacity = min_after_dequeue + 3 * self.batch_size
            batch = tf.train.batch(
                [image, label],
                batch_size=self.batch_size,
                num_threads=4,
                name='example_queue')
            # batch = tf.train.shuffle_batch(
            #    [image, label],
            #    batch_size=batch_size,
            #    num_threads=4,
            #    capacity=min_after_dequeue + 3 * batch_size,
            #    min_after_dequeue=min_after_dequeue,
            #    name='example_queue')
        return batch

    def next_batch(self):
        images, labels = self.sess.run(self.input_data)
        if self.aug:
            images_aug = seq.augment_images(images)
            images = images_aug
        """
        if self.aug_flag:
            images_aug = seq.augment_images(images)
            images = images_aug
            self.aug_flag = False
        else:
            self.aug_flag = True
        """
        one_hot_labels = np.zeros((self.batch_size, self.num_classes))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1

        return images, one_hot_labels
