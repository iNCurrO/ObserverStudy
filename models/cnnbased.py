from .DataSet.loaddata import *
from .DataSet.savedata import *
from .NeuralNet.convolution import *
import time


class cnnbased(object):
    def __init__(
            self, sess, img_size=65, batch_size=512, sample_num=100,
            dataset_name=['observer'], checkpoint_dir=None, sample_dir=None
    ):
        self._FLAGS = tf.app.flags.FLAGS
        self._sess = sess
        self._img_size = img_size

        self._sample_num = sample_num
        self._batch_size = batch_size

        self._dataset_name = dataset_name
        self._checkpoint_dir = checkpoint_dir
        self._sample_dir = sample_dir
        self._c_dim = 1
        self._dataset = loaddata(dataset_name, valrate=(1-0.05-self._FLAGS.datarate), testrate=0.05)
        # self._dataset = loaddata(dataset_name, valrate=0, testrate=1)
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
        self._network = self.network(image, repeatnum=self._FLAGS.depth, basechannel=self._FLAGS.basechannel)

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

    def resetdata(self, dataset_name, datarate=0.025):
        print("Set to data {}".format(dataset_name))
        self._dataset = loaddata(dataset_name, valrate=(1-0.05-datarate), testrate=0.05)

    def network(self, image, reuse=False, repeatnum=23, basechannel=64):
        with tf.variable_scope('network') as scope:
            if reuse:
                scope.reuse_variables()
            x = conv2d(image, basechannel, name='d_conv1', activation='lrelu', padding='VALID')
            for i in range(2, repeatnum):
                x = conv2d(x, basechannel, name='d_conv'+str(i), activation='lrelu', padding='VALID')
            x_temp = x
            result, w = fc(x, self._FLAGS.AFC, activation='linear', name='d_fc', debugging=True)
            w_temp = w[:, 0]
            plz_shape = x_temp.shape
            plz = tf.multiply(x_temp, tf.reshape(w_temp, [plz_shape[1], plz_shape[2], plz_shape[3]]))
            plz = tf.expand_dims(tf.reduce_sum(plz,3), -1)
            plz_min = -1
            plz_max = 1
            plz_0to1 = (plz-plz_min)/(plz_max-plz_min)
            plz_0to255 = tf.image.convert_image_dtype(plz_0to1, dtype=tf.uint8)
            tf.summary.image('plz', plz_0to255, max_outputs=1)
            return result

    def train(self, epoch_num=100, lr=1e-2, beta1=0.5, continued=False):

        tf.global_variables_initializer().run()
        counter = 1
        stopflag = True
        start_time = time.time()
        # Load model
        if continued:
            ckpt = tf.train.get_checkpoint_state(self._checkpoint_dir)
            self.saver.restore(self._sess, os.path.join(self._checkpoint_dir, os.path.basename(ckpt.model_checkpoint_path)))
        for epoch in range(epoch_num):
            while stopflag is True:
                counter += 1
                # Load batch
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
                # Training (calculate backward loss)
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
                    # print("Epoch: [{0:2d}] [{1:4d}/{2:4d}] time: {3:4.4f}, loss: {4:.8f}, accuracy: {5:3.3f}".format(
                    #     epoch, self._dataset.train.getposition, self._dataset.train.num_example,
                    #     time.time() - start_time,
                    #     loss, accuracy * 100
                    # ))
                    self.train_writer.add_summary(summary, counter)
                # Save point
                if np.mod(counter, 5000) == 2:
                    if not os.path.exists(self._checkpoint_dir):
                        os.makedirs(self._checkpoint_dir)
                    self.saver.save(self._sess, os.path.join(self._checkpoint_dir, "observer.model"),
                                    global_step=counter)

                if self._dataset.train.getposition == 0:
                    stopflag = False
                # Validation result
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
                    # print("Epoch: [{0:2d}] [Validation] time: {1:4.4f}, loss: {2:.8f}, accuracy: {3:3.3f}".format
                    #       (epoch, time.time() - start_time, loss, accuracy * 100)
                    #       )
                    self.test_writer.add_summary(summary, counter)
            # Validation result every epoch
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
            # print("Validation result for Epoch [{0:2d}] time: {1:4.4f}, loss: {2:.8f}, accuracy: {3: 3.3f} ".format
            #       (epoch, time.time() - start_time, loss, accuracy * 100)
            #       )
            stopflag = True
        # Validation results after all training done
        # stopflag = True
        # counter = 0
        # loss = 0
        # accuracy = 0
        # while stopflag is True:
        #     counter += 1
        #     if self._FLAGS.AFC is 2:
        #         img1, img2, batch_label = self._dataset.test.next_batch(self._sample_num, must_full=True)
        #         temploss, tempaccuracy = self._sess.run([
        #             self._loss, self._accuracy],
        #             feed_dict={
        #                 self.inputs1: img1, self.inputs2: img2,
        #                 self.labels: batch_label
        #             })
        #     else:
        #         img1, img2, img3, img4, batch_label = self._dataset.test.next_batch(self._sample_num, must_full=True)
        #         temploss, tempaccuracy = self._sess.run([
        #             self._loss, self._accuracy],
        #             feed_dict={
        #                 self.inputs1: img1, self.inputs2: img2, self.inputs3: img3, self.inputs4: img4,
        #                 self.labels: batch_label
        #             })
        #     loss += temploss
        #     accuracy += tempaccuracy
        #     print("[Test Result] time: {0:4.4f}, loss: {1:.8f}, accuracy: {2:3.3f}".format(
        #         time.time() - start_time, temploss, tempaccuracy * 100))
        #     if self._dataset.test.getposition == 0:
        #         stopflag = False
        # print("[Result] loss: {1:.8f}, accuracy: {2:3.3f}".format(
        #     time.time() - start_time, loss / counter, accuracy * 100 / counter))

        self.saver.save(self._sess, os.path.join(self._checkpoint_dir, "observer.model"), global_step=counter)

    def loadandsampling(self, withsave=False):
        ckpt = tf.train.get_checkpoint_state(self._checkpoint_dir)
        self.saver.restore(self._sess, os.path.join(self._checkpoint_dir, os.path.basename(ckpt.model_checkpoint_path)))
        counter = 0
        loss = 0
        accuracy = 0
        accuracy_list = []
        if withsave:
            tvars = tf.trainable_variables()
            tvars_vals = self._sess.run(tvars)
            for var, val in zip(tvars, tvars_vals):
                name = "{}.npy".format(self._checkpoint_dir+var.name.replace("/", "-").replace(".", "-").replace(":", "-"))
                np.save(name, val)
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
                temploss, tempaccuracy, summary= self._sess.run([
                    self._loss, self._accuracy, self.merged],
                    feed_dict={
                        self.inputs1: img1, self.inputs2: img2, self.inputs3: img3, self.inputs4: img4,
                        self.labels: batch_label
                    })
            self.after_writer.add_summary(summary, counter)
            loss += temploss
            accuracy += tempaccuracy
            accuracy_list += [tempaccuracy]
            # print("Validation result time: {0:4.4f}, loss: {1:.8f}, accuracy: {2: 3.3f}".format(
            #     time.time() - start_time, temploss, tempaccuracy * 100
            # ))
            if self._dataset.test.getposition == 0 or counter==10:
                stopflag = False
        print("[Test Result] time: {0:4.4f}, loss: {1:.8f}, accuracy: {2:3.3f}".format(
            time.time() - start_time, loss / counter, accuracy * 100 / counter))
        print(accuracy_list)

    # Load and sampling with activation map saving
    def loadandsampling_CAM(self, withsave=False):
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
        realcount = 0
        while stopflag is True:
            counter += 1
            img1, img2, img3, img4, batch_label = self._dataset.test.next_batch(1, must_full=True)
            temploss, tempaccuracy, summary= self._sess.run([
                self._loss, self._accuracy, self.merged],
                feed_dict={
                    self.inputs1: img1, self.inputs2: img2, self.inputs3: img2, self.inputs4: img2,
                    self.labels: batch_label
                })
            self.after_writer.add_summary(summary, counter)
            counter += 1
            temploss, tempaccuracy, summary= self._sess.run([
                self._loss, self._accuracy, self.merged],
                feed_dict={
                    self.inputs1: img2, self.inputs2: img2, self.inputs3: img2, self.inputs4: img2,
                    self.labels: batch_label
                })
            self.after_writer.add_summary(summary, counter)
            counter += 1
            temploss, tempaccuracy, summary= self._sess.run([
                self._loss, self._accuracy, self.merged],
                feed_dict={
                    self.inputs1: img3, self.inputs2: img2, self.inputs3: img2, self.inputs4: img2,
                    self.labels: batch_label
                })
            self.after_writer.add_summary(summary, counter)
            counter += 1
            temploss, tempaccuracy, summary= self._sess.run([
                self._loss, self._accuracy, self.merged],
                feed_dict={
                    self.inputs1: img4, self.inputs2: img2, self.inputs3: img2, self.inputs4: img2,
                    self.labels: batch_label
                })
            self.after_writer.add_summary(summary, counter)
            counter += 1
            temploss, tempaccuracy, summary= self._sess.run([
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
            realcount+=1
            if self._dataset.test.getposition == 0 or realcount==10:
                stopflag = False
        print("[Test Result] time: {0:4.4f}, loss: {1:.8f}, accuracy: {2:3.3f}".format(
            time.time() - start_time, loss / counter, accuracy * 100 / counter))
        print(accuracy_list)

