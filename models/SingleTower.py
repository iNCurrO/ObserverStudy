from .DataSet.loaddata import *
from .DataSet.savedata import *
from .NeuralNet.convolution import *
import time
from scipy import io


class STmodel(object):
    def __init__(
            self, sess, img_size=65, batch_size=256, sample_num=100,
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
            self._dataset = loaddata(dataset_name, valrate=0.01, testrate=0.01)
        # self._dataset = loaddata(dataset_name, valrate=0, testrate=1)
        self.inputs = tf.placeholder(
            tf.float32, [None, self._img_size, self._img_size, self._c_dim]
        )
        self.labels = tf.placeholder(
            tf.float16, [None, 2]
        )
        self._network = self.OneImageNetwork(self.inputs)
        t_vars = tf.trainable_variables()

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

    def resetdata(self, dataset_name, testrate=0.05):
        print("Set to data {}".format(dataset_name))
        self._dataset = loaddata(dataset_name, testrate=testrate)

    # def OneImageNetwork(self, image, reuse=False):  ## For CP 4~7
    #     with tf.variable_scope('network') as scope:
    #         if reuse:
    #             scope.reuse_variables()
    #         basechannel = 128
    #         h0_0 = conv2d(image, basechannel, name='conv0_0', activation='lrelu')
    #         h0_1 = conv2d(h0_0, basechannel, name='conv0_1', activation='lrelu')
    #         h0 = maxpool(h0_1, name='conv0_maxpool')
    #
    #         h1_0 = conv2d(h0, basechannel * 2, name='conv1_0', activation='lrelu')
    #         h1_1 = conv2d(h1_0, basechannel * 2, name='conv1_1', activation='lrelu')
    #         h1 = maxpool(h1_1, name='conv1_maxpool')
    #
    #         h2_0 = conv2d(h1, basechannel * 4, name='conv2_0', activation='lrelu')
    #         h2_1 = conv2d(h2_0, basechannel * 4, name='conv2_1', activation='lrelu')
    #         h2 = maxpool(h2_1, name='conv2_maxpool')
    #
    #         h4 = fc(h2, 1024, activation='lrelu', name='fc1', withdropout=True)
    #         h5 = fc(h4, 1024, activation='lrelu', name='fc2', withdropout=True)
    #         h6 = fc(h5, 2, activation='linear', name='fc3')
    #
    #         return h6

    def OneImageNetwork(self, image, reuse=False):
        with tf.variable_scope('network') as scope:
            if reuse:
                scope.reuse_variables()
            basechannel = 16
            h1 = conv2d(image, basechannel, name='d_conv1', activation='lrelu', padding='VALID')
            h2 = conv2d(h1, basechannel, name='d_conv2', activation='lrelu', padding='VALID')
            h3 = conv2d(h2, basechannel, name='d_conv3', activation='lrelu', padding='VALID')
            h4 = conv2d(h3, basechannel, name='d_conv4', activation='lrelu', padding='VALID')
            h5 = conv2d(h4, basechannel, name='d_conv5', activation='lrelu', padding='VALID')
            h6 = conv2d(h5, basechannel, name='d_conv6', activation='lrelu', padding='VALID')
            h7 = conv2d(h6, basechannel, name='d_conv7', activation='lrelu', padding='VALID')
            h8 = conv2d(h7, basechannel, name='d_conv8', activation='lrelu', padding='VALID')
            h9 = conv2d(h8, basechannel, name='d_conv9', activation='lrelu', padding='VALID')
            h10 = conv2d(h9, basechannel, name='d_conv10', activation='lrelu', padding='VALID')
            h11 = conv2d(h10, basechannel, name='d_conv11', activation='lrelu', padding='VALID')
            h12 = conv2d(h11, basechannel, name='d_conv12', activation='lrelu', padding='VALID')
            h13 = conv2d(h12, basechannel, name='d_conv13', activation='lrelu', padding='VALID')
            h14 = conv2d(h13, basechannel, name='d_conv14', activation='lrelu', padding='VALID')
            h15 = conv2d(h14, basechannel, name='d_conv15', activation='lrelu', padding='VALID')
            h16 = conv2d(h15, basechannel, name='d_conv16', activation='lrelu', padding='VALID')
            h17 = conv2d(h16, basechannel, name='d_conv17', activation='lrelu', padding='VALID')
            h18 = conv2d(h17, basechannel, name='d_conv18', activation='lrelu', padding='VALID')
            h19 = conv2d(h18, basechannel, name='d_conv19', activation='lrelu', padding='VALID')
            h20 = conv2d(h19, basechannel, name='d_conv20', activation='lrelu', padding='VALID')
            h21 = conv2d(h20, basechannel, name='d_conv21', activation='lrelu', padding='VALID')
            h22 = conv2d(h21, basechannel, name='d_conv22', activation='lrelu', padding='VALID')
            h23 = conv2d(h22, basechannel, name='d_conv23', activation='lrelu', padding='VALID')
            # h24 = conv2d(h23, basechannel, name='d_conv24', activation='lrelu', padding='VALID')
            # h25 = conv2d(h24, basechannel, name='d_conv25', activation='lrelu', padding='VALID')
            # h26 = conv2d(h25, basechannel, name='d_conv26', activation='lrelu', padding='VALID')
            # h27 = conv2d(h26, basechannel, name='d_conv27', activation='lrelu', padding='VALID')
            # h28 = conv2d(h27, basechannel, name='d_conv28', activation='lrelu', padding='VALID')
            # h29 = conv2d(h28, basechannel, name='d_conv29', activation='lrelu', padding='VALID')
            # h30 = conv2d(h29, basechannel, name='d_conv30', activation='lrelu', padding='VALID')
            # h31 = conv2d(h30, basechannel, name='d_conv31', activation='lrelu', padding='VALID')
            # h32 = conv2d(h31, basechannel, name='d_conv32', activation='lrelu', padding='VALID')
            h33 = fc(h23, 2, activation='linear', name='d_fc')
            return h33

    def train(self, epoch_num=50, lr=1e-5, beta1=0.9):
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
                        epoch, self._dataset.train.getposition, self._dataset.train.num_example,
                        time.time() - start_time,
                        loss, accuracy * 100
                    ))
                    self.train_writer.add_summary(summary, counter)

                if np.mod(counter, 1000) == 2:
                    if not os.path.exists(self._checkpoint_dir):
                        os.makedirs(self._checkpoint_dir)
                    self.saver.save(self._sess, os.path.join(self._checkpoint_dir, "observer.model"),
                                    global_step=counter)

                if self._dataset.train.getposition == 0:
                    stopflag = False

                if np.mod(counter, 200) == 1:
                    valdata, vallabel = self._dataset.val.next_batch(100)
                    loss, accuracy, summary = self._sess.run([
                        self._loss, self._accuracy, self.merged],
                        feed_dict={
                            self.inputs: valdata, self.labels: vallabel
                        })
                    print(
                        "VEpoch: [{0:2d}] [Validation] time: {1:4.4f}, loss: {2:.8f}, accuracy: {3:3.3f} ".format
                            (
                            epoch, time.time() - start_time, loss, accuracy * 100
                        )
                    )
                    self.test_writer.add_summary(summary, counter)

            valdata, vallabel = self._dataset.val.next_batch(100)
            loss, accuracy, summary = self._sess.run([
                self._loss, self._accuracy, self.merged],
                feed_dict={
                    self.inputs: valdata, self.labels: vallabel
                })
            print("Validation result for Epoch [{0:2d}] time: {1:4.4f}, loss: {2:.8f}, accuracy: {3: 3.3f} ".format
                (
                epoch, time.time() - start_time, loss, accuracy * 100
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
            img, batch_label = self._dataset.test.next_batch(self._sample_num, must_full=True)
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
            print("[Test Result] time: {0:4.4f}, loss: {1:.8f}, accuracy: {2:3.3f}".format(
                time.time() - start_time, temploss, tempaccuracy * 100))
            if self._dataset.test.getposition == 0:
                stopflag = False

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
        self._sample_dataset = loadsampledata(sample_dir, sampledatanum=testnum, labeldice=label_dice)

    def loadandlabelsampling(self, number=100):
        ckpt = tf.train.get_checkpoint_state(self._checkpoint_dir)
        self.saver.restore(self._sess, os.path.join(self._checkpoint_dir, os.path.basename(ckpt.model_checkpoint_path)))
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
