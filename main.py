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
    tf.app.flags.DEFINE_integer('AFC', 4, 'which AFC?')
    # optionlist = [('circle', 'ramp', 8, 10), ('circle', 'hann', 16, 6), ('circle', 'mix',16, 12), ('elipse', 'ramp', 8, 14), ('elipse', 'hann', 32 , 8), ('elipse', 'mix', 16, 10)]
    # optionlist = [('spiculated', 'ramp', 32, 14), ('spiculated', 'hann', 32, 10), ('spiculated', 'mix',16, 14)]
    optionlist = [('circle', 'ramp', 8, 10)]
    # optionlist = [('spiculated', trecon, twidth, tdepth) for trecon in ['mix'] for twidth in [8, 16, 32] for tdepth in [2,4,6,8,10,12,14] ]
    datarate= 0.5
    for (signalshape, filtername, tempwidth, temp) in optionlist:
        for batchnum in [1, 2, 4, 8, 16, 32, 64, 128]:
        # for datarate in [0.0025, 0.0125, 0.025, 0.05, 0.075, 0.125, 0.25, 0.75]:
            print('=======================================================================================')
            if 'depth' in list(tf.app.flags.FLAGS):
                delattr(tf.app.flags.FLAGS, 'depth')
                delattr(tf.app.flags.FLAGS, 'basechannel')
                delattr(tf.app.flags.FLAGS, 'datarate')
                delattr(tf.app.flags.FLAGS, 'batchnum')
                tf.reset_default_graph()
            tf.app.flags.DEFINE_integer('batchnum', batchnum, 'batch number')
            tf.app.flags.DEFINE_integer('basechannel', tempwidth, "basechannelNum")
            tf.app.flags.DEFINE_integer('depth', temp, 'repeat layer nums')
            tf.app.flags.DEFINE_float('datarate', datarate, 'datarate')
            with tf.Session(config=config) as sess:
                # ckdir = './floor' + str(FLAGS.depth) + '_ellipse_noPooling_channel' + str(FLAGS.basechannel) + '_ramp'
                # ckdir = './floor' + str(FLAGS.depth) + '_' + signalshape + '_shallow_channel' + str(
                #     FLAGS.basechannel) + '_' + filtername
                ckdir = './floor_batchnumcase_'+str(batchnum)
                if FLAGS.AFC is 2:
                    ckdir = ckdir + '_2AFC'
                else:
                    ckdir = ckdir + '_4AFC'
                if datarate is not 0.9:
                    ckdir = ckdir + '_datarate' + str(datarate)
                print(ckdir)
                start_time = time.time()
                if filtername is not 'mix':
                    datasetname = ['Observer_' + signalshape + '_trans_' + filtername]
                else:
                    datasetname = ['Observer_' + signalshape + '_trans_' + fn for fn in ['ramp', 'hann']]
                    delattr(tf.app.flags.FLAGS, 'datarate')
                    tf.app.flags.DEFINE_float('datarate', datarate/2, 'datarate')
                srcnn = STmodel(sess, checkpoint_dir=ckdir,
                                sample_dir=None, dataset_name=datasetname)
                print('!!{}sec to ready'.format(time.time()-start_time))
                srcnn.train()
                if filtername is not 'mix':
                    srcnn.loadandsampling()
                else:
                    for datasetnamelist in datasetname:
                        srcnn.resetdata([datasetnamelist])
                        srcnn.loadandsampling()
    # for signalshape in ['elipse', 'circle']:
    #     for filtername in ['ramp']:
    #         for tempwidth in [8, 16, 32]:
    #             for temp in [2, 4, 6, 8, 10, 12, 14]: # 32는 28부터 해야됭 >_<
    #                 for datarate in [0.15]:
    #                     print('=======================================================================================')
    #                     if 'depth' in list(tf.app.flags.FLAGS):
    #                         delattr(tf.app.flags.FLAGS, 'depth')
    #                         delattr(tf.app.flags.FLAGS, 'basechannel')
    #                         delattr(tf.app.flags.FLAGS, 'datarate')
    #                         tf.reset_default_graph()
    #                     tf.app.flags.DEFINE_integer('basechannel', tempwidth, "basechannelNum")
    #                     tf.app.flags.DEFINE_integer('depth', temp, 'repeat layer nums')
    #                     tf.app.flags.DEFINE_float('datarate', datarate, 'datarate')
    #                     with tf.Session(config=config) as sess:
    #                         # ckdir = './floor' + str(FLAGS.depth) + '_ellipse_noPooling_channel' + str(FLAGS.basechannel) + '_ramp'
    #                         ckdir = './floor' + str(FLAGS.depth) + '_'+signalshape+'_shallow_channel' + str(FLAGS.basechannel) + '_' + filtername
    #                         if FLAGS.AFC is 2:
    #                             ckdir = ckdir + '_2AFC'
    #                         else:
    #                             ckdir = ckdir + '_4AFC'
    #                         if datarate is not 0.9:
    #                             ckdir = ckdir + '_datarate' + str(datarate)
    #                         print(ckdir)
    #                         start_time = time.time()
    #                         if filtername is not 'mix':
    #                             datasetname = ['Observer_' + signalshape + '_trans_' + filtername]
    #                         else:
    #                             datasetname = ['Observer_' + signalshape + '_trans_' + fn for fn in ['ramp', 'hann']]
    #                         srcnn = STmodel(sess, checkpoint_dir=ckdir,
    #                                         sample_dir=None, dataset_name=datasetname)
    #                                         # sample_dir=None, dataset_name=['Observer_elipse_trans_ramp'])
    #                         # show_all_variables()
    #                         # print('!!{}sec to ready'.format(time.time()-start_time))
    #                         # srcnn.train()
    #                         if filtername is not 'mix':
    #                             srcnn.loadandsampling()
    #                         else:
    #                             for datasetnamelist in datasetname:
    #                                 srcnn.resetdata([datasetnamelist])
    #                                 srcnn.loadandsampling()


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
