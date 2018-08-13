import numpy
import scipy.misc
from random import shuffle
import tensorflow as tf

def imread(pathes):
    return numpy.array([(scipy.misc.imread(path) / 255) for path in pathes]).astype(numpy.float32)

def scrable(a, b, axis=0):
    c = numpy.random.random(a.shape[0:2])
    idx = numpy.argsort(c, axis=axis)
    shuffled_a = a[idx, numpy.arange(a.shape[1])[None, :], :, :]
    shuffled_b = b[numpy.arange(b.shape[0])[:, None], numpy.transpose(idx)]
    return shuffled_a, shuffled_b

# def readimage(fname):
# 	# queue = tf.train.string_input_producer(list(fname), shuffle=False, seed=None)
# 	# reader = tf.WholeFileReader()
# 	# filename, data = reader.read(queue)
# 	# image = tf.image.decode_png(data, channels=3)
# 	# image.set_shape([64, 64, 3])
# 	# image = tf.image.rgb_to_grayscale(image)
# 	# return tf.to_float(image)
# 	tempimg = Image.open(fname).convert('L')
# 	resultimg = np.asarray(tempimg, dtype=float)
# 	if tempimg.mode == 'L':
# 		resultimg = np.reshape(resultimg, [resultimg.shape[0], resultimg.shape[1], 1])
# 	else:
# 		resultimg = np.reshape(resultimg, [resultimg.shape[0], resultimg.shape[1], 3])
# 	tempimg.close()
# 	# print(np.min((resultimg/127.5 - 1.).astype(np.float32)))
# 	return (resultimg/255).astype(np.float32)

class DataSet(object):
    def __init__(self, images1, images2):
        # assert len(images1.shape) < 5,\
        # 	'From Loading data... Image Shape is strange ({}). please check. '.format(images1.shape)
        # assert images1.shape[0] == label.shape[0], "From Loading data... # of image and label differ please check" \
        #                                            "image {} : label {} ,".format(images1.shape[0], label.shape[0])
        self._num_examples = images1.shape[0]
        self._index_in_epoch = 0
        self._index_in_epoch2 = 0
        self._images1 = images1
        self._images2 = images2
        # self._labels = label
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images1 = self._images1[perm]
        # self._labels = self._labels[perm]
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images2 = self._images2[perm]

    @property
    def getimage(self):
        return self._images1, self._images2

    @property
    def getlabels(self):
        return self._labels

    @property
    def getposition(self):
        return self._index_in_epoch

    @property
    def num_example(self):
        return self._num_examples

    def next_batch(self, batch_size, fake_data=False, must_full=True):
        if tf.app.flags.FLAGS.AFC is 2:
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
            if self._index_in_epoch >= self._num_examples and must_full:
                start = self._num_examples - batch_size
                end = self._num_examples
                self._index_in_epoch = 0
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images1 = self._images1[perm]
                tempimages1 = imread(self._images1[start:end])[:, :, :, None]
                tempimages2 = imread(self._images2[start:end])[:, :, :, None]
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images2 = self._images2[perm]
            else:
                end = self._index_in_epoch
                tempimages1 = imread(self._images1[start:end])[:, :, :, None]
                tempimages2 = imread(self._images2[start:end])[:, :, :, None]
            tempimages = numpy.asarray([tempimages1, tempimages2])
            templabel = numpy.asarray([1, 0], dtype=numpy.int8)
            templabels = numpy.asarray([templabel for _ in range(batch_size)])
            tempimages, templabels = scrable(tempimages, templabels)
            return tempimages[0], tempimages[1], templabels
        else:
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
            if self._index_in_epoch >= self._num_examples and must_full:
                start = self._num_examples - batch_size
                end = self._num_examples
                self._index_in_epoch = 0
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images1 = self._images1[perm]
                tempimages1 = imread(self._images1[start:end])[:, :, :, None]
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images2 = self._images2[perm]
            else:
                end = self._index_in_epoch
                tempimages1 = imread(self._images1[start:end])[:, :, :, None]
            start = self._index_in_epoch2
            self._index_in_epoch2 += batch_size
            if self._index_in_epoch2 >= self._num_examples/3 and must_full:
                num_batched = int(numpy.floor(self._num_examples/3))
                start = num_batched - batch_size
                end = num_batched
                self._index_in_epoch2 = 0
                tempimages2, tempimages3, tempimages4= imread(self._images2[start:end])[:, :, :, None], \
                                                       imread(self._images2[start + num_batched:end + num_batched])[:, :, :, None], \
                                                       imread(self._images2[start + 2 * num_batched:end + 2 * num_batched])[:, :, :, None]
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images2 = self._images2[perm]
            else:
                end = self._index_in_epoch2
                num_batched = int(numpy.floor(self._num_examples/3))
                tempimages2, tempimages3, tempimages4= imread(self._images2[start:end])[:, :, :, None], \
                                                                    imread(self._images2[start + num_batched:end + num_batched])[:, :, :, None], \
                                                                    imread(self._images2[start + 2 * num_batched:end + 2 * num_batched])[:, :, :, None]
            tempimages = numpy.asarray([tempimages1, tempimages2, tempimages3, tempimages4])
            templabel = numpy.asarray([1, 0, 0, 0], dtype=numpy.int8)
            templabels = numpy.asarray([templabel for _ in range(batch_size)])
            tempimages, templabels = scrable(tempimages, templabels)
            return tempimages[0], tempimages[1], tempimages[2], tempimages[3], templabels
