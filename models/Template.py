from .DataSet.loaddata import *
from .DataSet.savedata import *
from .NeuralNet.convolution import *
import time


class DCGAN(object):
	def __init__(
			self, sess, img_size=64, batch_size=64, sample_num=64,
			dataset_name='observer', checkpoint_dir=None, sample_dir=None
	):
		self._sess = sess
		self._img_size = img_size

		self._sample_num = sample_num
		self._batch_size = batch_size

		self._dataset_name = dataset_name
		self._checkpoint_dir = checkpoint_dir
		self._sample_dir = sample_dir
		self._dataset = loaddata(self._dataset_name, valrate=0, testrate=0)
		self.inputs = tf.placeholder(
			tf.float32, [None, self._img_size, self._img_size, 1], name='real_images'
		)

		t_vars = tf.trainable_variables()
		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

		# self.saver = tf.train.Saver()

	def train(self, epoch_num=25, dataset='observer', lr=1e-5, beta1=0.5):

		tf.global_variables_initializer().run()

		counter = 1
		stopflag = True
		start_time = time.time()

		for epoch in range(epoch_num):
			while stopflag is True:
				batch_image1, batch_image2, batch_image3, batch_image4,\
				batch_label = self._dataset.train.next_batch(self._batch_size)

				if counter == 1:

					print("Epoch: [{0:2d}] [{1:4d}/{2:4d}] time: {3:4.4f}, d_loss: {4:.8f}, g_loss: {5:.8f}".format(
						-1, data.train.getposition, data.train.num_example, time.time() - start_time,
						err_d_fake + err_d_real, err_g
					))

				counter += 1
				print("Epoch: [{0:2d}] [{1:4d}/{2:4d}] time: {3:4.4f}, d_loss: {4:.8f}, g_loss: {5:.8f}".format(
					epoch, data.train.getposition, data.train.num_example, time.time() - start_time,
					err_d_fake + err_d_real, err_g
				))

				if np.mod(counter, 200) == 1:
					samples, g_loss, d_loss = self._sess.run([self.G, self.g_loss, self.d_loss], feed_dict={
						self.inputs: sample_inputs
					})

					print("[Sampling result] d_loss: {0:.8f}, g_loss: {1:.8f}".format(d_loss, g_loss))
					saveimg(congrateimg(samples, 8), 'train_{0:02d}_{1:04d}'.format(
						epoch, counter
					), isgrey=True)

				# if np.mod(counter, 500) == 2:
				# 	if not os.path.exists(self._checkpoint_dir):
				# 		os.makedirs(self._checkpoint_dir)
				# 	self.saver.save(self._sess, os.path.join(self._checkpoint_dir, "DCGAN.model"), global_step=counter)

				if data.train.getposition == 0:
					stopflag = False
			stopflag = True

