from .DataSet.loaddata import *
from .DataSet.savedata import *
from .NeuralNet.convolution import *
import time


class msObserver(object):
	def __init__(
			self, sess, img_size=64, batch_size=64, sample_num=64,
			dataset_name='observer', checkpoint_dir=None, sample_dir=None
	):
		self._FLAGS = tf.app.flags.FLAGS
		self._sess = sess
		self._img_size = img_size

		self._sample_num = sample_num
		self._batch_size = batch_size

		self._dataset_name = dataset_name
		self._checkpoint_dir = checkpoint_dir
		self._sample_dir = sample_dir
		self._c_dim = 13
		self._dataset = loaddata(dataset_name, valrate=(1-0.05-self._FLAGS.datarate), testrate=0.05)
		self.inputs1 = tf.placeholder(
            tf.float32, [None, self._img_size, self._img_size, self._c_dim]
        )
		self.inputs2 = tf.placeholder(
            tf.float32, [None, self._img_size, self._img_size, self._c_dim]
        )
		if self._FLAGS.AFC is not 2:
			self.inputs3 = tf.placeholder(
                tf.float32, [None, self._img_size, self._img_size, self._c_dim]
            )
			self.inputs4 = tf.placeholder(
                tf.float32, [None, self._img_size, self._img_size, self._c_dim]
            )
		t_vars = tf.trainable_variables()
		if self._FLAGS.AFC is 2:
			image = tf.concat([self.inputs1, self.inputs2], axis=3)
			self.labels = tf.placeholder(
                tf.float16, [None, 2]
            )
		else:
			image = tf.concat([self.inputs1, self.inputs2, self.inputs3, self.inputs4], axis=3)
			self.labels = tf.placeholder(
                tf.float16, [None, 4]
            )
		self._network = self.network(image)
		# self._network = self.network(image, repeatnum=self._FLAGS.depth, basechannel=self._FLAGS.basechannel)
		self._loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._network, labels=self.labels)
        )
		tf.summary.scalar("loss", self._loss)
		self._accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(tf.nn.softmax(self._network), 1), tf.argmax(tf.nn.softmax(self.labels), 1)),
                tf.float32
            )
        )
		tf.summary.scalar("accuracy", self._accuracy)
		self.train_writer = tf.summary.FileWriter(checkpoint_dir + '/log/train', sess.graph)
		self.test_writer = tf.summary.FileWriter(checkpoint_dir + '/log/test')
		self.after_writer = tf.summary.FileWriter(checkpoint_dir+'/log/after')
		self.merged = tf.summary.merge_all()
		self.saver = tf.train.Saver()


	def train(self, epoch_num=25, dataset='observer', lr=1e-5, beta1=0.5):
		optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(self._loss)
		tf.global_variables_initializer().run()
		counter = 1
		stopflag = True
		start_time = time.time()
		for epoch in range(epoch_num):
			while stopflag is True:
				counter += 1
				if self._FLAGS.AFC is 2:
					img1, img2, batch_label = self._dataset.train.next_batch(self._batch_size, must_full=True)
					self._sess.run(optim, feed_dict={
						self.inputs1: img1, self.inputs2: img2,
						self.labels: batch_label
					})
				else:
					img1, img2, img3, img4, batch_label = self._dataset.train.next_batch(self._batch_size, must_full=True)
					self._sess.run(optim, feed_dict={
						self.inputs1: img1, self.inputs2: img2, self.inputs3: img3, self.inputs4: img4,
						self.labels: batch_label
                    })
				if np.mod(counter, 10) == 1:
					if self._FLAGS.AFC is 2:
						loss, accuracy, summary = self._sess.run([self._loss, self._accuracy, self.merged],
                                                                 feed_dict={
                                                                     self.inputs1: img1, self.inputs2: img2,
                                                                     self.labels: batch_label
                                                                 })
					else:
						loss, accuracy, summary = self._sess.run([self._loss, self._accuracy, self.merged],
                                                                 feed_dict={
                                                                     self.inputs1: img1, self.inputs2: img2,
                                                                     self.inputs3: img3, self.inputs4: img4,
                                                                     self.labels: batch_label
                                                                 })
					print("Epoch: [{0:2d}] [{1:4d}/{2:4d}] time: {3:4.4f}, loss: {4:.8f}, accuracy: {5:3.3f}".format(
                        epoch, self._dataset.train.getposition, self._dataset.train.num_example,
                        time.time() - start_time,
                        loss, accuracy * 100
                    ))
					self.train_writer.add_summary(summary, counter)

				if np.mod(counter, 5000) == 2:
					if not os.path.exists(self._checkpoint_dir):
						os.makedirs(self._checkpoint_dir)
					self.saver.save(self._sess, os.path.join(self._checkpoint_dir, "observer.model"),
                                    global_step=counter)

				if self._dataset.train.getposition == 0:
					stopflag = False
				if np.mod(counter, 200) == 1:
					if self._FLAGS.AFC is 2:
						valdata1, valdata2, vallabel = self._dataset.val.next_batch(self._sample_num)
						loss, accuracy = self._sess.run([
                            self._loss, self._accuracy],
                            feed_dict={
                                self.inputs1: valdata1, self.inputs2: valdata2,
                                self.labels: vallabel
                            })
					else:
						valdata1, valdata2, valdata3, valdata4, vallabel = self._dataset.val.next_batch(self._sample_num)
						loss, accuracy, summary = self._sess.run([
                            self._loss, self._accuracy, self.merged],
                            feed_dict={
                                self.inputs1: valdata1, self.inputs2: valdata2, self.inputs3: valdata3,
                                self.inputs4: valdata4,
                                self.labels: vallabel
                            })
					print("Epoch: [{0:2d}] [Validation] time: {1:4.4f}, loss: {2:.8f}, accuracy: {3:3.3f}".format
                          (epoch, time.time() - start_time, loss, accuracy * 100)
                          )
					self.test_writer.add_summary(summary, counter)
			if self._FLAGS.AFC is 2:
				valdata1, valdata2, vallabel = self._dataset.val.next_batch(self._sample_num)
				loss, accuracy = self._sess.run([
                    self._loss, self._accuracy],
                    feed_dict={
                        self.inputs1: valdata1, self.inputs2: valdata2,
                        self.labels: vallabel
                    })
			else:
				valdata1, valdata2, valdata3, valdata4, vallabel = self._dataset.val.next_batch(self._sample_num)
				loss, accuracy = self._sess.run([
                    self._loss, self._accuracy],
                    feed_dict={
                        self.inputs1: valdata1, self.inputs2: valdata2, self.inputs3: valdata3, self.inputs4: valdata4,
                        self.labels: vallabel
                    })
			print("Validation result for Epoch [{0:2d}] time: {1:4.4f}, loss: {2:.8f}, accuracy: {3: 3.3f} ".format
                  (epoch, time.time() - start_time, loss, accuracy * 100)
                  )
			stopflag = True
		self.saver.save(self._sess, os.path.join(self._checkpoint_dir, "observer.model"), global_step=counter)

	def loadandsampling(self, withsave=False):
		ckpt = tf.train.get_checkpoint_state(self._checkpoint_dir)
		self.saver.restore(self._sess, os.path.join(self._checkpoint_dir, os.path.basename(ckpt.model_checkpoint_path)))
		counter = 0
		loss = 0
		accuracy = 0
		accuracy_list = []
		if withsave:
			weights = {}
			tvars = tf.trainable_variables()
			tvars_vals = self._sess.run(tvars)
			for var, val in zip(tvars, tvars_vals):
				weights[var.name] = val
			name = "{}.npy".format(self._checkpoint_dir)
			np.save(name, weights)
		start_time = time.time()
		stopflag = True
		while stopflag is True:
			counter += 1
			if self._FLAGS.AFC is 2:
				img1, img2, batch_label = self._dataset.test.next_batch(self._sample_num, must_full=True)
				temploss, tempaccuracy, summary = self._sess.run([
					self._loss, self._accuracy, self.merged],
					feed_dict={
						self.inputs1: img1, self.inputs2: img2,
						self.labels: batch_label
					})
			else:
				img1, img2, img3, img4, batch_label = self._dataset.test.next_batch(self._sample_num, must_full=True)
				temploss, tempaccuracy, summary = self._sess.run([
					self._loss, self._accuracy, self.merged],
					feed_dict={
						self.inputs1: img1, self.inputs2: img2, self.inputs3: img3, self.inputs4: img4,
						self.labels: batch_label
					})
			self.after_writer.add_summary(summary, counter)
			loss += temploss
			accuracy += tempaccuracy
			accuracy_list += [tempaccuracy]
			print("Validation result time: {0:4.4f}, loss: {1:.8f}, accuracy: {2: 3.3f}".format(
				time.time() - start_time, temploss, tempaccuracy * 100
			))
			if self._dataset.test.getposition == 0 or counter == 10:
				stopflag = False
		print("[Test Result] time: {0:4.4f}, loss: {1:.8f}, accuracy: {2:3.3f}".format(
			time.time() - start_time, loss / counter, accuracy * 100 / counter))
		print(accuracy_list)

