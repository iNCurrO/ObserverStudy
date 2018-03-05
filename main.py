
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
	with tf.Session() as sess:
		ckdir = './cp4'
		srcnn = STmodel(sess, checkpoint_dir=ckdir, sample_dir=None, dataset_name='observer')
		show_all_variables()
		srcnn.train()
		srcnn.loadandsampling()
		# srcnn = STmodel(sess, checkpoint_dir=ckdir, sample_dir=None, dataset_name='observerhann')
		# show_all_variables()
		# srcnn.loadandsampling()
		# srcnn = STmodel(sess, checkpoint_dir=ckdir, sample_dir=None, dataset_name='observershepp')
		# show_all_variables()
		# srcnn.loadandsampling()
		# srcnn = STmodel(sess, checkpoint_dir=ckdir, sample_dir=None, dataset_name='observerlongi')
		# show_all_variables()
		# srcnn.loadandsampling()
		# srcnn = STmodel(sess, checkpoint_dir=ckdir, sample_dir=None, dataset_name='observerlongihann')
		# show_all_variables()
		# srcnn.loadandsampling()
		# srcnn = STmodel(sess, checkpoint_dir=ckdir, sample_dir=None, dataset_name='observerlongishepp')
		# show_all_variables()
		# srcnn.loadandsampling()
		# srcnn = STmodel(sess, checkpoint_dir=ckdir, sample_dir=None, dataset_name='observer1mmtransramp')
		# show_all_variables()
		# srcnn.loadandsampling()
		# srcnn = STmodel(sess, checkpoint_dir=ckdir, sample_dir=None, dataset_name='observer1mmtranshann')
		# show_all_variables()
		# srcnn.loadandsampling()

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
	tf.app.run()
