from models.SingleTower import STmodel
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import time
from models.cnnbased import cnnbased

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def floatrange(start, end, step):
    r = start
    while r < end:
        yield r
        r += step
#


def mainwithtransferlearning(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.app.flags.DEFINE_integer('AFC', 4, 'which AFC?')
    optionlist = [('circle', 'ramp', 'circle', 'hann', 32, 8), ('circle', 'ramp', 'circle', 'hann', 8, 10),
                  ('circle', 'hann', 'circle', 'ramp', 32, 8), ('circle', 'hann', 'circle', 'ramp', 8, 10)]
    for (startshape, startfilter, endshape, endfilter, width, depth) in optionlist:
        print('=======================================================================================')
        if 'depth' in list(tf.app.flags.FLAGS):
            delattr(tf.app.flags.FLAGS, 'depth')
            delattr(tf.app.flags.FLAGS, 'basechannel')
            delattr(tf.app.flags.FLAGS, 'datarate')
            tf.reset_default_graph()
        tf.app.flags.DEFINE_integer('basechannel', width, "basechannelNum")
        tf.app.flags.DEFINE_integer('depth', depth, 'repeat layer nums')
        tf.app.flags.DEFINE_float('datarate', 0.15, 'datarate')
        with tf.Session(config=config) as sess:
            ckdir = './transferlearning_depth' + str(FLAGS.depth) +'_channel' + str(
                FLAGS.basechannel) + '_from' +  startshape + startfilter + '_to' + endshape + endfilter
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
            show_all_variables()
            model.train()
            if endfilter == 'white':
                datasetname = ['white']
            else:
                datasetname = ['Observer_' + endshape + '_trans_' + endfilter]
            model.resetdata(dataset_name=datasetname, datarate=0.01)
            model.loadandsampling()


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.app.flags.DEFINE_integer('AFC', 4, 'which AFC?')
    optionlist = [ ('circle', 'white', 16, 12)]
    for (signalshape, filtername, tempwidth, temp) in optionlist:
        for datarate in floatrange(0.05, 0.95, 0.05):
        # for datarate in [0.0025, 0.0125, 0.025, 0.05, 0.075, 0.125, 0.25, 0.75]:
            print('=======================================================================================')
            if 'depth' in list(tf.app.flags.FLAGS):
                delattr(tf.app.flags.FLAGS, 'depth')
                delattr(tf.app.flags.FLAGS, 'basechannel')
                delattr(tf.app.flags.FLAGS, 'datarate')
                tf.reset_default_graph()
            tf.app.flags.DEFINE_integer('basechannel', tempwidth, "basechannelNum")
            tf.app.flags.DEFINE_integer('depth', temp, 'repeat layer nums')
            tf.app.flags.DEFINE_float('datarate', datarate, 'datarate')
            with tf.Session(config=config) as sess:
                # ckdir = './msObserver'+ '_' + signalshape + '_depth' + str(FLAGS.depth) +'_channel' + str(
                #     FLAGS.basechannel) + '_' + filtername
                ckdir = './SLCNN_circle_white'
                if FLAGS.AFC is 2:
                    ckdir = ckdir + '_2AFC'
                else:
                    ckdir = ckdir + '_4AFC'
                if datarate is not 0.9:
                    ckdir = ckdir + '_datarate' + str(datarate)
                print(ckdir)
                start_time = time.time()
                if filtername == 'white':
                    datasetname = ['white']
                else:
                    datasetname = ['Observer_' + signalshape + '_trans_' + filtername]
                srcnn = STmodel(sess, checkpoint_dir=ckdir,
                                sample_dir=None, dataset_name=datasetname)
                print('!!{}sec to ready'.format(time.time()-start_time))
                show_all_variables()
                srcnn.train()
                srcnn.loadandsampling()
                # srcnn.loadandsampling_CAM()
    # for signalshape in ['elipse', 'circle']:
    #     for filtername in ['hann', 'mix']:
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
    #                         # print('!!{}sec to ready'.format(time.time()-start_time))
    #                         # show_all_variables()
    #                         # srcnn.train()
    #                         if filtername is not 'mix':
    #                             srcnn.loadandsampling()
    #                         else:
    #                             for datasetnamelist in datasetname:
    #                                 srcnn.resetdata([datasetnamelist])
    #                                 srcnn.loadandsampling()


if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.run()
