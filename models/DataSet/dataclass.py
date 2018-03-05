import numpy


class DataSet(object):
	def __init__(self, images1, images2, images3, images4, label):
		assert len(images1.shape) < 5,\
			'From Loading data... Image Shape is strange ({}). please check. '.format(images1.shape)
		assert images1.shape[0] == label.shape[0], "From Loading data... # of image and label differ please check" \
																 "image {} : label {} ,".format(images1.shape[0], label.shape[0])
		self._num_examples, self._image_width, self._image_height = images1.shape[0], images1.shape[1], images1.shape[2]
		self._index_in_epoch = 0
		perm = numpy.arange(self._num_examples)
		numpy.random.shuffle(perm)
		self._images1 = images1[perm]
		self._images2 = images2[perm]
		self._images3 = images3[perm]
		self._images4 = images4[perm]
		self._labels = label[perm]

	@property
	def getimage(self):
		return self._images1, self._images2, self._images3, self._images4

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
			self._images1, self._images2, self._images3, self._images4, self._labels = \
				self._images1[perm], self._images2[perm], self._images3[perm], self._images4[perm], self._labels[perm]
			tempimages1, tempimages2, tempimages3, tempimages4, templabels = \
				self._images1[start:end], self._images2[start:end], self._images3[start:end], self._images4[start:end],\
				self._labels[start:end]
			return tempimages1, tempimages2, tempimages3, tempimages4, templabels
		elif self._index_in_epoch >= self._num_examples:
			end = self._num_examples
			self._index_in_epoch = 0
			tempimages1, tempimages2, tempimages3, tempimages4, templabels = \
				self._images1[start:end], self._images2[start:end], self._images3[start:end], self._images4[start:end], \
				self._labels[start:end]
			perm = numpy.arange(self._num_examples)
			numpy.random.shuffle(perm)
			self._images1, self._images2, self._images3, self._images4, self._labels = \
				self._images1[perm], self._images2[perm], self._images3[perm], self._images4[perm], self._labels[perm]
			return tempimages1, tempimages2, tempimages3, tempimages4, templabels
		else:
			end = self._index_in_epoch
			return numpy.asarray(self._images1[start:end], dtype=numpy.float32),\
				numpy.asarray(self._images2[start:end], dtype=numpy.float32),\
				numpy.asarray(self._images3[start:end], dtype=numpy.float32),\
				numpy.asarray(self._images4[start:end], dtype=numpy.float32),\
				numpy.asarray(self._labels[start:end])
