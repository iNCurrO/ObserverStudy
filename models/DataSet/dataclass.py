import numpy
import scipy.misc


def imread(pathes):
    return numpy.array([(scipy.misc.imread(path) / 255) for path in pathes]).astype(numpy.float32)


class DataSet(object):
    def __init__(self, images, label):
        assert len(images.shape) < 5, \
            'From Loading data... Image Shape is strange ({}). please check. '.format(images.shape)
        assert images.shape[0] == label.shape[0], "From Loading data... # of image and label differ please check" \
                                                  "image {} : label {} ,".format(images.shape[0], label.shape[0])
        self._num_examples = images.shape[0]
        self._index_in_epoch = 0
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = images[perm]
        self._labels = label[perm]

    @property
    def getimage(self):
        return imread(self._images)[:, :, :, None]

    @property
    def getlabels(self):
        return self._labels

    @property
    def getposition(self):
        return self._index_in_epoch

    @property
    def num_example(self):
        return self._num_examples

    def next_batch(self, batch_size, fake_data=False, must_full=False):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch >= self._num_examples and must_full:
            start = self._num_examples - batch_size
            end = self._num_examples
            self._index_in_epoch = 0
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images, self._labels = self._images[perm], self._labels[perm]
            tempimages, templabels = imread(self._images[start:end])[:, :, :, None], self._labels[start: end]
            return tempimages, templabels
        elif self._index_in_epoch >= self._num_examples:
            end = self._num_examples
            self._index_in_epoch = 0
            tempimages, templabels = imread(self._images[start:end])[:, :, :, None], self._labels[start: end]
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images, self._labels = self._images[perm], self._labels[perm]
            return tempimages, templabels
        else:
            end = self._index_in_epoch
            tempimages, templabels = imread(self._images[start:end])[:, :, :, None], self._labels[start: end]
            return tempimages, templabels
