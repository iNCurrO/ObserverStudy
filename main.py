from models.SingleTower import STmodel
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import time
from models.cnnbased import cnnbased

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def floatrange(start, end, step):
    r = start
    while r < end:
        yield r
        r += step
#


def main(_):#withtransferlearning(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.app.flags.DEFINE_integer('AFC', 4, 'which AFC?')
    optionlist = [('circle', 'hann', 'spiculated', 'hann', 32, 8)]
    # hyperoption = [(lr, bn) for lr in [0.00002, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1] for bn in [2, 4, 8, 16, 32, 64]]
    for (startshape, startfilter, endshape, endfilter, width, depth) in optionlist:
        for datarate in [1,2,3,4,5,6,7]:
            print('=======================================================================================')
            if 'depth' in list(tf.app.flags.FLAGS):
                delattr(tf.app.flags.FLAGS, 'depth')
                delattr(tf.app.flags.FLAGS, 'basechannel')
                delattr(tf.app.flags.FLAGS, 'datarate')
                tf.reset_default_graph()
            tf.app.flags.DEFINE_integer('basechannel', width, "basechannelNum")
            tf.app.flags.DEFINE_integer('depth', depth, 'repeat layer nums')
            tf.app.flags.DEFINE_float('datarate', 0.0573591, 'datarate')
            # tf.app.flags.DEFINE_float('datarate', 0.0375*datarate, 'datarate')
            with tf.Session(config=config) as sess:
                ckdir = './transferlearning_depth' + str(FLAGS.depth) +'_channel' + str(
                    FLAGS.basechannel) + '_from' +  startshape + startfilter + '_to' + endshape + endfilter+'_datarate'+str(datarate)
                ckdir += 'yes'
                if FLAGS.AFC is 2:
                    ckdir = ckdir + '_2AFC'
                else:
                    ckdir = ckdir + '_4AFC'
                print(ckdir)
                start_time = time.time()
                if startfilter == 'white':
                    datasetname = ['white']
                else:
                    datasetname = ['Observer_' + startshape + '_trans_' + startfilter]
                model = cnnbased(sess, checkpoint_dir=ckdir,
                                sample_dir=None, dataset_name=datasetname)
                print('!!{}sec to ready'.format(time.time()-start_time))
                # show_all_variables()
                # model.train()
                # model.loadandsampling()
                if endfilter == 'white':
                    datasetname = ['white']
                else:
                    datasetname = ['Observer_' + endshape + '_trans_' + endfilter]
                model.resetdata(dataset_name=datasetname, datarate=0.0375*datarate)
                # model.loadandsampling()
                # model.train(continued=True)
                print('After tf---------------------------------------')
                model.loadandsampling()


if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.run()
