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
    tf.app.flags.DEFINE_integer('AFC', 2, 'which AFC?')
    for tempwidth in [16, 32, 128]:
        for temp in [20, 22, 26, 28, 30]:
            if 'depth' in list(tf.app.flags.FLAGS):
                delattr(tf.app.flags.FLAGS, 'depth')
                delattr(tf.app.flags.FLAGS, 'basechannel')
                tf.reset_default_graph()
            tf.app.flags.DEFINE_integer('basechannel', tempwidth, "basechannelNum")
            tf.app.flags.DEFINE_integer('depth', temp, 'repeat layer nums')
            with tf.Session(config=config) as sess:
                ckdir = './floor' + str(FLAGS.depth) + '_ellipse_noPooling_channel' + str(FLAGS.basechannel) + '_mk2'
                if FLAGS.AFC is 2:
                    ckdir = ckdir + '_2AFC'
                print(ckdir)
                start_time = time.time()
                srcnn = STmodel(sess, checkpoint_dir=ckdir,
                                # sample_dir=None, dataset_name=['observer1mmtranshann', 'observer1mmtransramp'])
                                sample_dir=None, dataset_name=['Observer_elipse_trans_ramp', 'Observer_elipse_trans_hann'])
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
                # srcnn = STmodel(sess, checkpoint_dir=ckdir, sample_dir=None, dataset_name=['Observer_elipse_trans_ramp'])
                # show_all_variables()
                # srcnn.loadandsampling()
                # srcnn.resetdata(dataset_name=['observer1mmtranshann'])
                # srcnn.loadandsampling()
                # srcnn.resetdata(dataset_name=['observer1mmtransramp'])
                # srcnn.loadandsampling()
                srcnn.resetdata(dataset_name=['Observer_elipse_trans_ramp'])
                srcnn.loadandsampling()
                srcnn.resetdata(dataset_name=['Observer_elipse_trans_hann'])
                srcnn.loadandsampling()


if __name__ == '__main__':
    # for basechannel in [16, 32, 64, 128]:
    #     FLAGS = tf.app.flags.FLAGS
    #     tf.app.flags.DEFINE_integer('basechannel', basechannel, "basechannelNum")
    #     tf.app.flags.DEFINE_integer('depth', 24, 'repeat layer nums')
    #     tf.app.run
    FLAGS = tf.app.flags.FLAGS
    tf.app.run()
# FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_integer('basechannel', 64, "basechannelNum")
    # tf.app.flags.DEFINE_integer('depth', 23, 'repeat layer nums')
    # tf.app.run()
