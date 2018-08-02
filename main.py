from models.SingleTower import STmodel
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


#


def main(_):
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        ckdir = './FORONE_ellipse_floor' + str(FLAGS.depth) + '_noPooling_channel' + str(FLAGS.basechannel)
        srcnn = STmodel(sess, checkpoint_dir=ckdir,
                        # dataset_name=['observer1mmtranshann', 'observer1mmtransramp'])
                        dataset_name=['Observer_elipse_trans_ramp', 'Observer_elipse_trans_hann'])
        show_all_variables()
        srcnn.train()

        srcnn.resetsampledata(sample_dir=['Observer_elipse_trans_ramp'], label_dice=1)
        srcnn.loadandlabelsampling()
        srcnn.resetsampledata(sample_dir=['Observer_elipse_trans_ramp'], label_dice=2)
        srcnn.loadandlabelsampling()
        srcnn.resetsampledata(sample_dir=['Observer_elipse_trans_hann'], label_dice=1)
        srcnn.loadandlabelsampling()
        srcnn.resetsampledata(sample_dir=['Observer_elipse_trans_hann'], label_dice=2)
        srcnn.loadandlabelsampling()


#
# def main(_):
# 	with tf.Session() as sess:
# 		srcnn = STmodel(sess, checkpoint_dir='./cp', sample_dir=None, dataset_name='observer')
# 		show_all_variables()

# def main(_):
# 	with tf.Session() as sess:
# 		dcgan = DCGAN(sess, checkpoint_dir='./checkpoint', sample_dir=None)
#
# 		show_all_variables()
# 		# dcgan.train(finetune=True)
# 		dcgan.loadandsampling()
# 		dcgan.juststore()


if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('basechannel', 128, "basechannel Number")
    tf.app.flags.DEFINE_integer('depth', 24, 'repeat layer nums')
    tf.app.run()
