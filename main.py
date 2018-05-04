
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
	with tf.Session() as sess:
		ckdir = './cp2'
		start_time = time.time()
		srcnn = STmodel(sess, checkpoint_dir=ckdir,
							sample_dir=None, dataset_name=['observer1mmtranshann', 'observer1mmtransramp'])
		show_all_variables()
		print('!!{}sec to ready'.format(time.time()-start_time))
		srcnn.train()
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
		# srcnn.resetdata(dataset_name=['observer1mmtransramp'])
		# srcnn.loadandsampling()
		# srcnn.resetdata(dataset_name=['observer1mmtranshann'])
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
