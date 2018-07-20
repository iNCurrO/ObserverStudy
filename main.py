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
        ckdir = './FORONE_floor' + str(FLAGS.depth) + '_noPooling_channel' + str(FLAGS.basechannel)
        srcnn = STmodel(sess, checkpoint_dir=ckdir,
                        dataset_name=['observer1mmtranshann', 'observer1mmtransramp'])
        show_all_variables()
        srcnn.train()
        # 		srcnn.resetdata(dataset_name=['observer2mmtransramp'])
        # 		srcnn.loadandsampling()
        # 		srcnn.resetdata(dataset_name=['observer2mmtranshann'])
        # 		srcnn.loadandsampling()
        # 		# srcnn = STmodel(sess, checkpoint_dir=ckdir, sample_dir=None, dataset_name='observerlongi')
        # 		# show_all_variables()
        # 		# srcnn.loadandsampling()
        # 		# srcnn = STmodel(sess, checkpoint_dir=ckdir, sample_dir=None, dataset_name='observerlongihann')
        # 		# show_all_variables()
        # 		# srcnn.loadandsampling()
        # 		# srcnn = STmodel(sess, checkpoint_dir=ckdir,
        # 		# 					sample_dir=None, dataset_name=['observer1mmtransramp'])
        # 		srcnn.resetdata(dataset_name=['observer1mmtransramp'])
        # 		srcnn.loadandsampling()
        # 		srcnn.resetdata(dataset_name=['observer1mmtranshann'])
        # 		srcnn.loadandsampling()

        # srcnn.resetsampledata(sample_dir=['observer1mmtransramp'], label_dice=1)
        # srcnn.loadandlabelsampling()
        # srcnn.resetsampledata(sample_dir=['observer1mmtransramp'], label_dice=2)
        # srcnn.loadandlabelsampling()
        # srcnn.resetsampledata(sample_dir=['observer1mmtransramp'], label_dice=3)
        # srcnn.loadandlabelsampling()
        # srcnn.resetsampledata(sample_dir=['observer1mmtransramp'], label_dice=4)
        # srcnn.loadandlabelsampling()
        # srcnn.resetsampledata(sample_dir=['observer1mmtranshann'], label_dice=1)
        # srcnn.loadandlabelsampling()
        # srcnn.resetsampledata(sample_dir=['observer1mmtranshann'], label_dice=2)
        # srcnn.loadandlabelsampling()
        # srcnn.resetsampledata(sample_dir=['observer1mmtranshann'], label_dice=3)
        # srcnn.loadandlabelsampling()
        # srcnn.resetsampledata(sample_dir=['observer1mmtranshann'], label_dice=4)
        # srcnn.loadandlabelsampling()
        # srcnn.resetsampledata(sample_dir=['observer2mmtransramp'], label_dice=1)
        # srcnn.loadandlabelsampling()
        # srcnn.resetsampledata(sample_dir=['observer2mmtransramp'], label_dice=2)
        # srcnn.loadandlabelsampling()
        # srcnn.resetsampledata(sample_dir=['observer2mmtransramp'], label_dice=3)
        # srcnn.loadandlabelsampling()
        # srcnn.resetsampledata(sample_dir=['observer2mmtransramp'], label_dice=4)
        # srcnn.loadandlabelsampling()
        # srcnn.resetsampledata(sample_dir=['observer2mmtranshann'], label_dice=1)
        # srcnn.loadandlabelsampling()
        # srcnn.resetsampledata(sample_dir=['observer2mmtranshann'], label_dice=2)
        # srcnn.loadandlabelsampling()
        # srcnn.resetsampledata(sample_dir=['observer2mmtranshann'], label_dice=3)
        # srcnn.loadandlabelsampling()
        # srcnn.resetsampledata(sample_dir=['observer2mmtranshann'], label_dice=4)
        # srcnn.loadandlabelsampling()


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
    tf.app.flags.DEFINE_integer('basechannel', 16, "basechannel Number")
    tf.app.flags.DEFINE_integer('depth', 24, 'repeat layer nums')
    tf.app.run()
