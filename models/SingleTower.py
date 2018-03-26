from .DataSet.loaddata import *
from .DataSet.savedata import *
from .NeuralNet.convolution import *
import time
from scipy import io


class STmodel(object):
	def __init__(
			self, sess, img_size=65, batch_size=256, sample_num=64,
			dataset_name=[], checkpoint_dir=None, sample_dir=[], label_dice=1
	):
		self._sess = sess
		self._img_size = img_size

		self._sample_num = sample_num
		self._batch_size = batch_size

		self._dataset_name = dataset_name
		self._checkpoint_dir = checkpoint_dir
		self._labeldice = label_dice
		if len(sample_dir) != 0:
			self._sample_dir = sample_dir
			self._sample_dataset = loadsampledata(sample_dir, labeldice=label_dice)
		self._c_dim = 1
		if len(dataset_name) != 0:
			self._dataset = loaddata(dataset_name, valrate=0.45, testrate=0.05)
			# self._dataset = loaddata(dataset_name, valrate=0, testrate=1)
		self.inputs = tf.placeholder(
			tf.float32, [None, self._img_size, self._img_size, self._c_dim]
		)
		self.labels = tf.placeholder(
			tf.float16, [None, 2]
		)
		self._network = self.OneImageNetwork(self.inputs)
		# self._network = self.network(self.inputs1, self.inputs2, self.inputs3, self.inputs4)
		t_vars = tf.trainable_variables()

		self._loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._network, labels=self.labels)
		)
		tf.summary.scalar("loss", self._loss)
		self._accuracy = tf.reduce_mean(
			tf.cast(
				tf.equal(tf.argmax(tf.nn.softmax(self._network), 1), tf.argmax(tf.nn.softmax(self.labels), 1)), tf.float32
			)
		)
		tf.summary.scalar("accuracy", self._accuracy)
		self.train_writer = tf.summary.FileWriter(checkpoint_dir + '/log/train', sess.graph)
		self.test_writer = tf.summary.FileWriter(checkpoint_dir+'/log/test')
		self.merged = tf.summary.merge_all()
		self.saver = tf.train.Saver()

	def resetdata(self, dataset_name, testrate=0.05):
		print("Set to data {}".format(dataset_name))
		self._dataset = loaddata(dataset_name, testrate=testrate)

	def OneImageNetwork(self, image, reuse=False): ## For CP 4~7
		with tf.variable_scope('network') as scope:
			if reuse:
				scope.reuse_variables()
			basechannel = 128
			h0_0 = conv2d(image, basechannel, name='conv0_0', activation='lrelu')
			h0_1 = conv2d(h0_0, basechannel, name='conv0_1', activation='lrelu', withbatch=False)
			h0 = maxpool(h0_1, name='conv0_maxpool')

			h1_0 = conv2d(h0, basechannel*2, name='conv1_0', activation='lrelu', withbatch=False)
			h1_1 = conv2d(h1_0, basechannel*2, name='conv1_1', activation='lrelu', withbatch=False)
			h1 = maxpool(h1_1, name='conv1_maxpool')

			h2_0 = conv2d(h1, basechannel*4, name='conv2_0', activation='lrelu', withbatch=False)
			h2_1 = conv2d(h2_0, basechannel*4, name='conv2_1', activation='lrelu', withbatch=False)
			h2 = maxpool(h2_1, name='conv2_maxpool')

			h3_0 = conv2d(h2, basechannel*8, name='conv3_0', activation='lrelu', withbatch=False)
			h3_1 = conv2d(h3_0, basechannel*8, name='conv3_1', activation='lrelu', withbatch=False)
			h3 = maxpool(h3_1, name='conv3_maxpool')

			h4 = fc(h3, 1024, activation='lrelu', name='fc1', withdropout=True)
			h5 = fc(h4, 1024, activation='lrelu', name='fc2', withdropout=True)
			h6 = fc(h5, 2, activation='linear', name='fc3')

			return h6



	# def network(self, img1, img2, img3, img4, reuse=False):
	# 	with tf.variable_scope('network') as scope:
	# 		if reuse:
	# 			scope.reuse_variables()
	# 		tempcon1 = tf.concat([img1, img2], axis=1)
	# 		tempcon2 = tf.concat([img3, img4], axis=1)
	# 		image = tf.concat([tempcon1, tempcon2], axis=2)
	# 		basechannel = 32
	# 		h0_0, w0_0 = conv2d(image, basechannel, name='d_conv0_0', activation='lrelu', withbatch=False, withweight=True)
	# 		h0_0 = tf.concat([h0_0, image], axis=3)
	# 		# tf.summary.image('w0_0', tf.reshape(w0_0, [8, 8, 1, 1]))
	# 		h0_1 = conv2d(h0_0, basechannel*2, name='d_conv0_1', activation='lrelu', withbatch=True)
	# 		h0 = tf.concat([h0_1, h0_0], axis=3)
	# 		h0_pool = maxpool(h0, k=3, s=2, name='d_conv0_maxpool')
	#
	# 		h1_0 = conv2d(h0_pool, basechannel*4, name='d_conv1_0', activation='lrelu', withbatch=True)
	# 		h1_0 = tf.concat([h1_0, h0_pool], axis=3)
	# 		h1_1 = conv2d(h1_0, basechannel*8, name='d_conv1_1', activation='lrelu', withbatch=True)
	# 		h1 = tf.concat([h1_1, h1_0], axis=3)
	# 		h1_pool = maxpool(h1, k=3, s=2, name='d_conv1_maxpool')
	#
	# 		h2_0 = conv2d(h1_pool, basechannel*16, name='d_conv2_0', activation='lrelu', withbatch=True)
	# 		# h2_0 = tf.concat([h2_0, h1_pool], axis=3)
	# 		h2_1 = conv2d(h2_0, basechannel*16, name='d_conv2_1', activation='lrelu', withbatch=True)
	# 		h2 = tf.concat([h2_1, h2_0], axis=3)
	# 		h2_pool = maxpool(h2, k=3, s=2, name='d_conv2_maxpool')
	#
	# 		# h3_0 = conv2d(h2_pool, basechannel*64, name='d_conv3_0', activation='lrelu', withbatch=False)
	# 		# h3_0 = tf.concat([h3_0, h2_pool], axis=3)
	# 		# h3_1 = conv2d(h3_0, basechannel*64, name='d_conv3_1', activation='lrelu', withbatch=False)
	# 		# h3 = tf.concat([h3_1, h3_0], axis=3)
	# 		# h3_pool = maxpool(h3, k=3, s=2, name='d_conv3_maxpool')
	#
	# 		h4 = fc(h2_pool, 1024, activation='lrelu', name='d_fc1')
	# 		h4_drop = tf.nn.dropout(h4, keep_prob=0.5)
	# 		h5 = fc(h4_drop, 4, activation='linear', name='d_fc2')
	# 		h5_drop = tf.nn.dropout(h5, keep_prob=0.5)
	# 		return h5_drop


	# def network(self, img1, img2, img3, img4, reuse=False):
	# 	with tf.variable_scope('network') as scope:
	# 		if reuse:
	# 			scope.reuse_variables()
	# 		# tempcon1 = tf.concat([img1, img2], axis=1)
	# 		# tempcon2 = tf.concat([img3, img4], axis=1)
	# 		print(img1.shape)
	# 		image = tf.concat([img1, img2, img3, img4], axis=3)
	# 		basechannel = 256
	# 		h0 = conv2d(image, basechannel, k=9, name='d_conv0', activation='lrelu')
	# 		h0_pool = maxpool(h0, k=5, s=2, name='d_conv0_maxpool')
	# 		h1 = conv2d(h0_pool, basechannel, k=9, name='d_conv1', activation='lrelu')
	# 		h1_pool = maxpool(h1, k=5, s=2, name='d_conv1_maxpool')
	# 		#
	# 		# h1_0 = conv2d(h0_pool, basechannel*2, name='d_conv1_0', activation='lrelu')
	# 		# h1_1 = conv2d(h1_0, basechannel*2, name='d_conv1_1', activation='lrelu')
	# 		# h1_pool = maxpool(h1_1, k=5, s=2, name='d_conv1_maxpool')
	#
	# 		# h2_0 = conv2d(h1_pool, basechannel*4, name='d_conv2_0', activation='lrelu')
	# 		# h2_1 = conv2d(h2_0, basechannel*4, name='d_conv2_1', activation='lrelu')
	# 		# h2_pool = maxpool(h2_1, k=5, s=2, name='d_conv2_maxpool')
	#
	# 		# h3_0 = conv2d(h2_pool, basechannel*8, name='d_conv3_0', activation='lrelu')
	# 		# h3_1 = conv2d(h3_0, basechannel*8, name='d_conv3_1', activation='lrelu')
	# 		# h3_pool = maxpool(h3_1, k=3, s=2, name='d_conv3_maxpool')
	#
	# 		# h4 = fc(h1_pool, 1024, activation='lrelu', name='d_fc_1')
	# 		# h4 = tf.nn.dropout(h4, keep_prob=0.5)
	# 		h5 = fc(h1_pool, 1024, activation='lrelu', name='d_fc_2', withdropout=True)
	# 		h6 = fc(h5, 4, activation='linear', name='d_fc_3')
	# 		return h6

	# def network(self, img1, img2, img3, img4, reuse=False):
	# 	with tf.variable_scope('network') as scope:
	# 		if reuse:
	# 			scope.reuse_variables()
	# 		tempcon1 = tf.concat([img1, img2], axis=1)
	# 		tempcon2 = tf.concat([img3, img4], axis=1)
	# 		image = tf.concat([tempcon1, tempcon2], axis=2)
	# 		# image = tf.concat([img1, img2, img3, img4], axis=3)
	# 		print(image.shape)
	# 		basechannel = 16
	# 		h0_0 = conv2d(image, basechannel, k=45, name='d_conv0_0', activation='lrelu', withbatch=False)
	# 		# h0_1 = conv2d(h0_0, basechannel, name='d_conv0_1', activation='lrelu')
	# 		h0_pool = avgpool(h0_0, k=2, s=2, name='d_conv0_maxpool')
	# 		#
	# 		h1_0 = conv2d(h0_pool, basechannel, k=19, name='d_conv1_0', activation='lrelu', withbatch=False)
	# 		# h1_1 = conv2d(h1_0, basechannel, name='d_conv1_1', activation='lrelu')
	# 		h1_pool = avgpool(h1_0, k=2, s=2, name='d_conv1_maxpool')
	# 		#
	# 		# h2_0 = conv2d(h1_pool, basechannel*4, name='d_conv2_0', activation='lrelu')
	# 		# h2_1 = conv2d(h2_0, basechannel*4, name='d_conv2_1', activation='lrelu')
	# 		# h2_pool = maxpool(h2_1, k=2, s=2, name='d_conv2_maxpool')
	#
	# 		# h4 = fc(h2_pool, 1024, activation='lrelu', name='d_fc_1', withdropout=True)
	# 		h5 = fc(h1_pool, 64, activation='lrelu', name='d_fc_2', withdropout=True)
	# 		h6 = fc(h5, 4, activation='linear', name='d_fc_3')
	# 		return h6

	def train(self, epoch_num=10, lr=1e-4, beta1=0.9):
		optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(self._loss)
		tf.global_variables_initializer().run()

		counter = 1
		stopflag = True
		start_time = time.time()

		for epoch in range(epoch_num):
			while stopflag is True:
				counter += 1
				img, batch_label = self._dataset.train.next_batch(self._batch_size, must_full=True)
				self._sess.run(optim, feed_dict={
					self.inputs: img, self.labels: batch_label
				})
				if np.mod(counter, 10) == 1:
					loss, accuracy, summary = self._sess.run([self._loss, self._accuracy, self.merged],
						feed_dict={
						self.inputs: img, self.labels: batch_label
					})
					print("Epoch: [{0:2d}] [{1:4d}/{2:4d}] time: {3:4.4f}, loss: {4:.8f}, accuracy: {5:3.3f}".format(
						epoch, self._dataset.train.getposition, self._dataset.train.num_example, time.time() - start_time,
						loss, accuracy*100
					))
					self.train_writer.add_summary(summary, counter)

				if np.mod(counter, 1000) == 2:
					if not os.path.exists(self._checkpoint_dir):
						os.makedirs(self._checkpoint_dir)
					self.saver.save(self._sess, os.path.join(self._checkpoint_dir, "observer.model"), global_step=counter)

				if self._dataset.train.getposition == 0:
					stopflag = False

			valdata, vallabel = self._dataset.val.next_batch(100)
			loss, accuracy, summary= self._sess.run([
				self._loss, self._accuracy, self.merged],
				feed_dict={
					self.inputs: valdata, self.labels: vallabel
				})
			# loss = self._loss.eval({
			#
			# })
			# accuracy = self._accuracy.eval({
			# 	self.inputs1: valdata1, self.inputs2: valdata2, self.inputs3: valdata3, self.inputs4: valdata4,
			# 	self.labels: vallabel
			# })
			print("Validation result for Epoch [{0:2d}] time: {1:4.4f}, loss: {2:.8f}, accuracy: {3: 3.3f} ".format
					(
					epoch, time.time() - start_time, loss, accuracy*100
					)
			)
			self.test_writer.add_summary(summary, counter)
			stopflag = True
		stopflag = True
		counter = 0
		loss = 0
		accuracy = 0
		while stopflag is True:
			counter += 1
			img, batch_label = self._dataset.test.next_batch(self._batch_size, must_full=True)
			temploss = self._loss.eval({
				self.inputs: img, self.labels: batch_label
			})
			tempaccuracy = self._accuracy.eval({
				self.inputs: img,	self.labels: batch_label
			})
			loss += temploss
			accuracy += tempaccuracy
			if self._dataset.test.getposition == 0:
				stopflag = False
		print("[Test Result] time: {0:4.4f}, loss: {1:.8f}, accuracy: {2:3.3f}".format(
			time.time() - start_time, loss/counter, accuracy * 100/counter))

		self.saver.save(self._sess, os.path.join(self._checkpoint_dir, "observer.model"), global_step=counter)

	def loadandsampling(self):
		ckpt = tf.train.get_checkpoint_state(self._checkpoint_dir)
		self.saver.restore(self._sess, os.path.join(self._checkpoint_dir, os.path.basename(ckpt.model_checkpoint_path)))
		counter = 0
		loss = 0
		accuracy = 0
		start_time = time.time()
		stopflag = True
		while stopflag is True:
			counter += 1
			img, batch_label = self._dataset.test.next_batch(100, must_full=True)
			temploss = self._loss.eval({
				self.inputs: img,
				self.labels: batch_label
			})
			tempaccuracy = self._accuracy.eval({
				self.inputs: img,
				self.labels: batch_label
			})
			loss += temploss
			accuracy += tempaccuracy
			print("Validation result time: {0:4.4f}, loss: {1:.8f}, accuracy: {2: 3.3f}".format(
				time.time() - start_time, temploss, tempaccuracy * 100
			))
			if self._dataset.test.getposition == 0:
				stopflag = False
		print("[Test Result] time: {0:4.4f}, loss: {1:.8f}, accuracy: {2:3.3f}".format(
			time.time() - start_time, loss / counter, accuracy * 100 / counter))

	def resetsampledata(self, sample_dir=[], testnum=1000, label_dice=2):
		print("Set to data {} (dice : {})".format(sample_dir[0], label_dice))
		self._labeldice = label_dice
		self._sample_dir = sample_dir
		self._sample_dataset = loadsampledata(sample_dir, sampledatanum=testnum,labeldice=label_dice)

	def loadandlabelsampling(self, number=100):
		ckpt = tf.train.get_checkpoint_state(self._checkpoint_dir)
		self.saver.restore(self._sess, os.path.join(self._checkpoint_dir, os.path.basename(ckpt.model_checkpoint_path)))
		loss = 0
		accuracy = 0
		start_time = time.time()
		img = self._sample_dataset.getimage
		batch_label = self._sample_dataset.getlabels
		loss, accuracy, label = self._sess.run([self._loss, self._accuracy, self._network],
			feed_dict={
				self.inputs: img,
				self.labels: batch_label
			})
		print("[Sampling Result] time: {0:4.4f}, loss: {1:.8f}, accuracy: {2:3.3f}".format(
			time.time() - start_time, loss, accuracy * 100))
		io.savemat(self._sample_dir[0] + str(self._labeldice), {'label': label})
