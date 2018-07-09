from models.SingleTower import STmodel
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


#


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # ckdir='./floor_init_channel16_gabor'
        # ckdir = './floor_combo1_floor4_k3_channe]' \
        # 		  'l16_plusOne_fc128'
        ckdir = './floor33_noPooling_channel16_mk2'
        start_time = time.time()
        srcnn = STmodel(sess, checkpoint_dir=ckdir,
                        sample_dir=None, dataset_name=['observer1mmtranshann', 'observer1mmtransramp'])
        show_all_variables()
        print('!!{}sec to ready'.format(time.time()-start_time))
        srcnn.train()
        # srcnn.loadandsampling()
        # srcnn.resetdata(dataset_name=['observer2mmtransramp'])
        # srcnn.loadandsampling()
        # srcnn.resetdata(dataset_name=['observer2mmtranshann'])
        # srcnn.loadandsampling()
        # srcnn = STmodel(sess, checkpoint_dir=ckdir, sample_dir=None, dataset_name='observerlongi')
        # show_all_variables()
        # srcnn.loadandsampling()
        # srcnn = STmodel(sess, checkpoint_dir=ckdir, sample_dir=None, dataset_name=['observer1mmtransramp'])
        # show_all_variables()
        # srcnn.loadandsampling()
        srcnn.resetdata(dataset_name=['observer1mmtransramp'])
        srcnn.loadandsampling()
        srcnn.resetdata(dataset_name=['observer1mmtranshann'])
        srcnn.loadandsampling()


if __name__ == '__main__':
    for basechannel in [16, 32, 64, 128]:
        FLAGS = tf.app.flags.FLAGS
        tf.app.flags.DEFINE_integer('basechannel', basechannel, "basechannelNum")
        tf.app.flags.DEFINE_integer('depth', 24, 'repeat layer nums')
        tf.app.run()
    for depth in range(20, 32, 2):
        FLAGS = tf.app.flags.FLAGS
        tf.app.flags.DEFINE_integer('basechannel', 64, "basechannelNum")
        tf.app.flags.DEFINE_integer('depth', depth, 'repeat layer nums')
        tf.app.run()
