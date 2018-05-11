from .DataSet.loaddata import *
from .DataSet.savedata import *
from .NeuralNet.convolution import *
import time


class STmodel(object):
	def __init__(
			self, sess, img_size=65, batch_size=256, sample_num=100,
			dataset_name=['observer'], checkpoint_dir=None, sample_dir=None
	):
		self._sess = sess
		self._img_size = img_size

		self._sample_num = sample_num
		self._batch_size = batch_size

		self._dataset_name = dataset_name
		self._checkpoint_dir = checkpoint_dir
		self._sample_dir = sample_dir
		self._c_dim = 1
		self._dataset = loaddata(dataset_name, valrate=0.05, testrate=0.05)
		# self._dataset = loaddata(dataset_name, valrate=0, testrate=1)
		self.inputs1 = tf.placeholder(
			tf.float32, [None, self._img_size, self._img_size, self._c_dim]
		)
		self.inputs2 = tf.placeholder(
			tf.float32, [None, self._img_size, self._img_size, self._c_dim]
		)
		self.inputs3 = tf.placeholder(
			tf.float32, [None, self._img_size, self._img_size, self._c_dim]
		)
		self.inputs4 = tf.placeholder(
			tf.float32, [None, self._img_size, self._img_size, self._c_dim]
		)
		self.labels = tf.placeholder(
			tf.float16, [None, 4]
		)
		self._network = self.network(self.inputs1, self.inputs2, self.inputs3, self.inputs4)
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

	def resetdata(self, dataset_name, testrate=0.025):
		print("Set to data {}".format(dataset_name))
		self._dataset = loaddata(dataset_name, testrate=testrate)

	# def network(self, img1, img2, img3, img4, reuse=False):
	# 	with tf.variable_scope('network') as scope:
	# 		if reuse:
	# 			scope.reuse_variables()
	# 		# tempcon1 = tf.concat([img1, img2], axis=1)
	# 		# tempcon2 = tf.concat([img3, img4], axis=1)
	# 		# image = tf.concat([tempcon1, tempcon2], axis=2)
	# 		image = tf.concat([img1, img2, img3, img4], axis=3)
	# 		basechannel = 16
	# 		h0_0, w = conv2d(image, basechannel, k=65, name='d_conv0_0', activation='linear', withbatch=False,
	# 				withweight=True, padding='VALID', isprepared=True)
	# 		x_min = tf.reduce_min(w)
	# 		x_max = tf.reduce_max(w)
	# 		w_0to1 = (w-x_min) / (x_max-x_min)
	# 		w_0to255 = tf.image.convert_image_dtype(w_0to1, dtype=tf.uint8)
	# 		w_trans = tf.transpose(w_0to255, [3, 0, 1, 2])
	# 		w1, w2, w3, w4 = tf.split(w_trans, 4, axis=3)
	# 		tf.summary.image('filters1', w1, max_outputs=basechannel)
	# 		tf.summary.image('filters2', w2, max_outputs=basechannel)
	# 		tf.summary.image('filters3', w3, max_outputs=basechannel)
	# 		tf.summary.image('filters4', w4, max_outputs=basechannel)
	#
	# 		# fc2 = fc(h0_0, 128, activation='linear', name='d_fc1', withdropout=True)
	# 		h5 = fc(h0_0, 4, activation='linear', name='d_fc2', withdropout=False)
	# 		return h5


	def network(self, img1, img2, img3, img4, reuse=False):
		with tf.variable_scope('network') as scope:
			if reuse:
				scope.reuse_variables()
			# tempcon1 = tf.concat([img1, img2], axis=1)
			# tempcon2 = tf.concat([img3, img4], axis=1)
			# image = tf.concat([tempcon1, tempcon2], axis=2)
			print(img1.shape)
			image = tf.concat([img1, img2, img3, img4], axis=3)
			basechannel = 8
			h0_0 = conv2d(image, basechannel, name='d_conv0_0', activation='lrelu')
			# h0_1 = conv2d(h0_0, basechannel, name='d_conv0_1', activation='lrelu', withbatch=False)
			# h0_2 = conv2d(h0_1, basechannel, name='d_conv0_2', activation='lrelu', withbatch=False)
			h0_pool = avgpool(h0_0, k=5, s=2, name='d_conv0_maxpool')

			# h1_0 = conv2d(h0_pool, basechannel*2, name='d_conv1_0', activation='lrelu')
			# h1_1 = conv2d(h1_0, basechannel*2, name='d_conv1_1', activation='lrelu', withbatch=False)
			# h1_2 = conv2d(h1_1, basechannel*2, name='d_conv1_2', activation='lrelu', withbatch=False)
			# h1_pool = avgpool(h1_2, k=5, s=2, name='d_conv1_maxpool')
			#
			# h2_0 = conv2d(h1_pool, basechannel*4, name='d_conv2_0', activation='lrelu')
			# h2_1 = conv2d(h2_0, basechannel*4, name='d_conv2_1', activation='lrelu', withbatch=False)
			# h2_2 = conv2d(h2_1, basechannel*4, name='d_conv2_2', activation='lrelu', withbatch=False)
			# h2_pool = avgpool(h2_2, k=5, s=2, name='d_conv2_maxpool')
			# #
			# h3_0 = conv2d(h2_pool, basechannel*8, name='d_conv3_0', activation='lrelu')
			# h3_1 = conv2d(h3_0, basechannel*8, name='d_conv3_1', activation='lrelu', withbatch=False)
			# h3_2 = conv2d(h3_1, basechannel*8, name='d_conv3_2', activation='lrelu', withbatch=False)
			# h3_pool = avgpool(h3_2, k=5, s=2, name='d_conv3_maxpool')

			# h4 = fc(h1_pool, 1024, activation='lrelu', name='d_fc_1')
			# h4 = tf.nn.dropout(h4, keep_prob=0.5)
			h5 = fc(h0_pool, 256, activation='lrelu', name='d_fc_2', withdropout=True)
			h6 = fc(h5, 4, activation='linear', name='d_fc_3')
			return h6

	def train(self, epoch_num=50, lr=1e-3, beta1=0.5):
		optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(self._loss)
		tf.global_variables_initializer().run()

		counter = 1
		stopflag = True
		start_time = time.time()
		# if continued == True:
		# ckpt = tf.train.get_checkpoint_state(self._checkpoint_dir)
		# self.saver.restore(self._sess, os.path.join(self._checkpoint_dir, os.path.basename(ckpt.model_checkpoint_path)))
		for epoch in range(epoch_num):
			while stopflag is True:
				counter += 1
				img1, img2, img3, img4, batch_label = self._dataset.train.next_batch(self._batch_size, must_full=True)
				self._sess.run(optim, feed_dict={
					self.inputs1: img1, self.inputs2: img2, self.inputs3: img3, self.inputs4: img4, self.labels: batch_label[:, 0]
				})
				if np.mod(counter, 10) == 1:
					loss, accuracy, summary = self._sess.run([self._loss, self._accuracy, self.merged],
							feed_dict={
						self.inputs1: img1, self.inputs2: img2, self.inputs3: img3, self.inputs4: img4,
						self.labels: batch_label[:, 0]
					})
					print("Epoch: [{0:2d}] [{1:4d}/{2:4d}] time: {3:4.4f}, loss: {4:.8f}, accuracy: {5:3.3f}".format(
						epoch, self._dataset.train.getposition, self._dataset.train.num_example, time.time() - start_time,
						loss, accuracy*100
					))
					self.train_writer.add_summary(summary, counter)

				if np.mod(counter, 5000) == 2:
					if not os.path.exists(self._checkpoint_dir):
						os.makedirs(self._checkpoint_dir)
					self.saver.save(self._sess, os.path.join(self._checkpoint_dir, "observer.model"), global_step=counter)

				if self._dataset.train.getposition == 0:
					stopflag = False
				if np.mod(counter, 200) == 1:
					valdata1, valdata2, valdata3, valdata4, vallabel = self._dataset.val.next_batch(self._sample_num)
					loss, accuracy, summary = self._sess.run([
						self._loss, self._accuracy, self.merged],
							feed_dict={
								self.inputs1: valdata1, self.inputs2: valdata2, self.inputs3: valdata3, self.inputs4: valdata4,
								self.labels: vallabel
							})
					print("Epoch: [{0:2d}] [Validation] time: {1:4.4f}, loss: {2:.8f}, accuracy: {3:3.3f}".format
							(epoch, time.time() - start_time, loss, accuracy * 100)
					)
					self.test_writer.add_summary(summary, counter)

			valdata1, valdata2, valdata3, valdata4, vallabel = self._dataset.val.next_batch(self._sample_num)
			loss, accuracy = self._sess.run([
				self._loss, self._accuracy],
				feed_dict={
					self.inputs1: valdata1, self.inputs2: valdata2, self.inputs3: valdata3, self.inputs4: valdata4,
					self.labels: vallabel
				})
			print("Validation result for Epoch [{0:2d}] time: {1:4.4f}, loss: {2:.8f}, accuracy: {3: 3.3f} ".format
					(epoch, time.time() - start_time, loss, accuracy*100)
			)
			stopflag = True
		stopflag = True
		counter = 0
		loss = 0
		accuracy = 0
		while stopflag is True:
			counter += 1
			img1, img2, img3, img4, batch_label = self._dataset.test.next_batch(self._sample_num, must_full=True)
			temploss = self._loss.eval({
				self.inputs1: img1, self.inputs2: img2, self.inputs3: img3, self.inputs4: img4,
				self.labels: batch_label[:, 0]
			})
			tempaccuracy = self._accuracy.eval({
				self.inputs1: img1, self.inputs2: img2, self.inputs3: img3, self.inputs4: img4,
				self.labels: batch_label[:, 0]
			})
			loss += temploss
			accuracy += tempaccuracy
			print("[Test Result] time: {0:4.4f}, loss: {1:.8f}, accuracy: {2:3.3f}".format(
					time.time() - start_time, temploss, tempaccuracy * 100))
			if self._dataset.test.getposition == 0:
				stopflag = False
		print("[Result] loss: {1:.8f}, accuracy: {2:3.3f}".format(
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
			img1, img2, img3, img4, batch_label = self._dataset.test.next_batch(self._sample_num, must_full=True)
			temploss = self._loss.eval({
				self.inputs1: img1, self.inputs2: img2, self.inputs3: img3, self.inputs4: img4,
				self.labels: batch_label
			})
			tempaccuracy = self._accuracy.eval({
				self.inputs1: img1, self.inputs2: img2, self.inputs3: img3, self.inputs4: img4,
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

		def Newnetwork(self, img1, img2, img3, img4, reuse=False):
			def double(img1, img2, name='0'):
				with tf.variable_scope(name):
					image = tf.concat([img1, img2], axis=3)
					basechannel = 32
					h0 = conv2d(image, basechannel, name='d_conv0', activation='lrelu', withbatch=True)
					h0_pool = maxpool(h0, k=5, s=2, name='d_conv0_maxpool')

					h1 = conv2d(h0_pool, basechannel * 2, name='d_conv1', activation='lrelu', withbatch=True)
					h1_pool = maxpool(h1, k=5, s=2, name='d_conv1_maxpool')

					h2 = conv2d(h1_pool, basechannel * 4, name='d_conv2', activation='lrelu', withbatch=True)
					h2_pool = maxpool(h2, k=5, s=2, name='d_conv2_maxpool')

					h3 = fc(h2_pool, basechannel, name='d_fc')
					return h3

			def single(img1):
				basechannel = 32
				h0 = conv2d(img1, basechannel, name='s_conv0', activation='lrelu', withbatch=True)
				h0_pool = maxpool(h0, k=5, s=2, name='s_conv0_maxpool')

				h1 = conv2d(h0_pool, basechannel * 2, name='s_conv1', activation='lrelu', withbatch=True)
				h1_pool = maxpool(h1, k=5, s=2, name='s_conv1_maxpool')

				h2 = conv2d(h1_pool, basechannel * 4, name='s_conv2', activation='lrelu', withbatch=True)
				h2_pool = maxpool(h2, k=5, s=2, name='s_conv2_maxpool')

				h3 = fc(h2_pool, basechannel, name='s_fc')
				return h3

			a = single(img1)
			print(a.shape)
			a = tf.concat([a, double(img1, img2, name='1')], axis=1)
			a = tf.concat([a, double(img1, img3, name='2')], axis=1)
			a = tf.concat([a, double(img1, img4, name='3')], axis=1)
			a = tf.concat([a, double(img2, img3, name='4')], axis=1)
			a = tf.concat([a, double(img2, img4, name='5')], axis=1)
			a = tf.concat([a, double(img3, img4, name='6')], axis=1)
			h4 = fc(a, 1024, name='fc1')
			h5 = fc(h4, 1, name='fc2', activation='linear')
			return h5
