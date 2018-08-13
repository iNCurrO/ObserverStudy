from .DataSet.loaddata import *
from .DataSet.savedata import *
from .NeuralNet.convolution import *
import time


class STmodel(object):
    def __init__(
            self, sess, img_size=65, batch_size=128, sample_num=100,
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
        self._dataset = loaddata(dataset_name, valrate=0.05, testrate=0.05)
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
        # self._network = self.network(self.inputs1, self.inputs2, self.inputs3, self.inputs4)
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
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    def resetdata(self, dataset_name, testrate=0.025):
        print("Set to data {}".format(dataset_name))
        self._dataset = loaddata(dataset_name, testrate=testrate)

    def network(self, image, reuse=False, repeatnum=23, basechannel=64):
        with tf.variable_scope('network') as scope:
            if reuse:
                scope.reuse_variables()
            x = conv2d(image, basechannel, name='d_conv1', activation='lrelu', padding='VALID')
            for i in range(2, repeatnum):
                x = conv2d(x, basechannel, name='d_conv'+str(i), activation='lrelu', padding='VALID')
            result = fc(x, self._FLAGS.AFC, activation='linear', name='d_fc')
            return result



    def train(self, epoch_num=5, lr=1e-4, beta1=0.5):
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
                    # print("Epoch: [{0:2d}] [{1:4d}/{2:4d}] time: {3:4.4f}, loss: {4:.8f}, accuracy: {5:3.3f}".format(
                    #     epoch, self._dataset.train.getposition, self._dataset.train.num_example,
                    #     time.time() - start_time,
                    #     loss, accuracy * 100
                    # ))
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
                temploss, tempaccuracy = self._sess.run([
                    self._loss, self._accuracy],
                    feed_dict={
                        self.inputs1: img1, self.inputs2: img2,
                        self.labels: batch_label
                    })
            else:
                img1, img2, img3, img4, batch_label = self._dataset.test.next_batch(self._sample_num, must_full=True)
                temploss, tempaccuracy = self._sess.run([
                    self._loss, self._accuracy],
                    feed_dict={
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


    # def network(selfself, img1, img2, img3, img4, reuse=False):
    #     def bottleneck(inputb, basechannel=12, name='bottlenect'):
    #         with tf.variable_scope(name):
    #             x_b = batch_norm(inputb, name='firstBN')
    #             x_b = act_func(x_b, activation='relu')
    #             x_b = conv2d(x_b, 4*basechannel, k=1, activation='linear', name='firstConv')
    #             x_b = batch_norm(x_b, name='secondBN')
    #             x_b = act_func(x_b, activation='relu')
    #             x_b = conv2d(x_b, basechannel, k=3, activation='linear', name='secondConv')
    #             return x_b
    #
    #     def transition(inputt, basechannel=12, name='transition'):
    #         with tf.variable_scope(name):
    #             x_t = batch_norm(inputt)
    #             x_t = act_func(x_t, activation='relu')
    #             x_t = conv2d(x_t, basechannel, k=1, activation='linear')
    #             x_t = avgpool(x_t, k=2, s=2)
    #             return x_t
    #
    #     def denseblock(inputd, nb_layers, basechannel= 12, name='denseblock'):
    #         with tf.variable_scope(name):
    #             layerslist = list()
    #             layerslist.append(inputd)
    #             x_ = bottleneck(inputd, basechannel=basechannel)
    #             layerslist.append(x_)
    #             for i_ in range(nb_layers - 1):
    #                 x_ = concat(layerslist, axis=3)
    #                 x_ = bottleneck(x_, name='bottle_N_'+str(i_+1), basechannel=basechannel)
    #                 layerslist.append(x_)
    #             x_ = concat(layerslist, axis=3)
    #             return x_
    #     with tf.variable_scope('network') as scope:
    #         image = tf.concat([img1, img2, img3, img4], axis=3)
    #         basechannel = 40
    #         x = conv2d(image, output_dim=basechannel * 2, k=7, s=2, name='first_conv', activation='linear')
    #         x = maxpool(x, k=3, s=2)
    #         x = denseblock(x, nb_layers=6, basechannel=basechannel, name='denseblock_6')
    #         x = transition(x, name='trans_6', basechannel=basechannel)
    #         x = denseblock(x, nb_layers=12, basechannel=basechannel, name='denseblock_12')
    #         x = transition(x, name='trans_12', basechannel=basechannel)
    #         x = denseblock(x, nb_layers=64, basechannel=basechannel, name='denseblock_24')
    #         x = transition(x, name='trans_64', basechannel=basechannel)
    #         x = denseblock(x, nb_layers=48, name='denseblock_final')
    #         x = batch_norm(x)
    #         x = act_func(x, activation='relu')
    #         x = GAPool(x)
    #         result = fc(x, 4, activation='linear')
    #         return result


    # def network(self, img1, img2, img3, img4, reuse=False):
    # 	with tf.variable_scope('network') as scope:
    # 		if reuse:
    # 			scope.reuse_variables()
    # 		# tempcon1 = tf.concat([img1, img2], axis=1)
    # 		# tempcon2 = tf.concat([img3, img4], axis=1)
    # 		# image = tf.concat([tempcon1, tempcon2], axis=2)
    # 		image = tf.concat([img1, img2, img3, img4], axis=3)
    # 		basechannel = 16
    # 		h0_0, w = conv2d(image, basechannel, k=65, name='d_conv0_0', activation='lrelu', withbatch=False,
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

    # def network(self, img1, img2, img3, img4, reuse=False):
    #     with tf.variable_scope('network') as scope:
    #         if reuse:
    #             scope.reuse_variables()
    #         # tempcon1 = tf.concat([img1, img2], axis=1)
    #         # tempcon2 = tf.concat([img3, img4], axis=1)
    #         # image = tf.concat([tempcon1, tempcon2], axis=2)
    #         print(img1.shape)
    #         image = tf.concat([img1, img2, img3, img4], axis=3)
    #         basechannel = 16
    #         h0_0 = conv2d(image, basechannel, name='d_conv0_0', activation='lrelu')
    #         # h0_1 = conv2d(h0_0, basechannel, name='d_conv0_1', activation='lrelu', withbatch=False)
    #         # h0_2 = conv2d(h0_1, basechannel, name='d_conv0_2', activation='lrelu', withbatch=False)
    #         h0_pool = avgpool(h0_0, k=5, s=2, name='d_conv0_maxpool')
    #
    #         h1_0 = conv2d(h0_pool, basechannel * 2, name='d_conv1_0', activation='lrelu')
    #         # h1_1 = conv2d(h1_0, basechannel*2, name='d_conv1_1', activation='lrelu', withbatch=False)
    #         # h1_2 = conv2d(h1_1, basechannel*2, name='d_conv1_2', activation='lrelu', withbatch=False)
    #         h1_pool = avgpool(h1_0, k=5, s=2, name='d_conv1_maxpool')
    #         #
    #         h2_0 = conv2d(h1_pool, basechannel * 3, name='d_conv2_0', activation='lrelu')
    #         # h2_1 = conv2d(h2_0, basechannel*4, name='d_conv2_1', activation='lrelu', withbatch=False)
    #         # h2_2 = conv2d(h2_1, basechannel*4, name='d_conv2_2', activation='lrelu', withbatch=False)
    #         h2_pool = avgpool(h2_0, k=5, s=2, name='d_conv2_maxpool')
    #         # #
    #         h3_0 = conv2d(h2_pool, basechannel * 4, name='d_conv3_0', activation='lrelu')
    #         # h3_1 = conv2d(h3_0, basechannel*8, name='d_conv3_1', activation='lrelu', withbatch=False)
    #         # h3_2 = conv2d(h3_1, basechannel*8, name='d_conv3_2', activation='lrelu', withbatch=False)
    #         h3_pool = avgpool(h3_0, k=5, s=2, name='d_conv3_maxpool')
    #
    #         # h4 = fc(h1_pool, 1024, activation='lrelu', name='d_fc_1')
    #         # h4 = tf.nn.dropout(h4, keep_prob=0.5)
    #         h5 = fc(h3_pool, 128, activation='lrelu', name='d_fc_2', withdropout=True)
    #         h6 = fc(h5, 4, activation='linear', name='d_fc_3')
    #         return h6
