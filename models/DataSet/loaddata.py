from .dataclass import DataSet
from PIL import Image
import tensorflow as tf
import glob
import os
import numpy as np
from scipy import io
from random import shuffle


def readimage(fname):
    # queue = tf.train.string_input_producer(list(fname), shuffle=False, seed=None)
    # reader = tf.WholeFileReader()
    # filename, data = reader.read(queue)
    # image = tf.image.decode_png(data, channels=3)
    # image.set_shape([64, 64, 3])
    # image = tf.image.rgb_to_grayscale(image)
    # return tf.to_float(image)
    tempimg = Image.open(fname).convert('L')
    resultimg = np.asarray(tempimg, dtype=float)
    if tempimg.mode == 'L':
        resultimg = np.reshape(resultimg, [resultimg.shape[0], resultimg.shape[1], 1])
    else:
        resultimg = np.reshape(resultimg, [resultimg.shape[0], resultimg.shape[1], 3])
    tempimg.close()
    # print(np.min((resultimg/127.5 - 1.).astype(np.float32)))
    return (resultimg / 255).astype(np.float32)


def loaddata(dataname, valrate=0.2, testrate=0.1, dir_='D:\CTgit\Image'):
    assert (valrate <= 1.0) & (testrate <= 1.0) & (valrate + testrate <= 1.0), 'From loaddata ' \
                                                                               'Validation rate or testrate can\'t over 1'
    data_dir_ = []
    label = np.zeros([4, 0])
    for dataset in dataname:
        if dataset == 'observer2mmtransramp':
            data_dir_temp = glob.glob(os.path.join(dir_, 'Observer_2mm_trans_ramp', '*'))
            mat_file = io.loadmat(os.path.join(dir_, 'Observer_2mm_trans_ramp', 'label.mat'))
            data_dir_temp = [d for d in data_dir_temp if os.path.isdir(d)]
        elif dataset == 'observer2mmtranshann':
            data_dir_temp = glob.glob(os.path.join(dir_, 'Observer_2mm_trans_hann', '*'))
            mat_file = io.loadmat(os.path.join(dir_, 'Observer_2mm_trans_hann', 'label.mat'))
            data_dir_temp = [d for d in data_dir_temp if os.path.isdir(d)]
        elif dataset == 'observer2mmlongiramp':
            data_dir_temp = glob.glob(os.path.join(dir_, 'Observer_2mm_longi_ramp', '*'))
            data_dir_temp = [d for d in data_dir_temp if os.path.isdir(d)]
        elif dataset == 'observer2mmlongihann':
            data_dir_temp = glob.glob(os.path.join(dir_, 'Observer_2mm_longi_hann', '*'))
            data_dir_temp = [d for d in data_dir_temp if os.path.isdir(d)]
        elif dataset == 'observer1mmtransramp':
            data_dir_temp = glob.glob(os.path.join(dir_, 'Observer_1mm_trans_ramp', '*'))
            mat_file = io.loadmat(os.path.join(dir_, 'Observer_1mm_trans_ramp', 'label.mat'))
            data_dir_temp = [d for d in data_dir_temp if os.path.isdir(d)]
        elif dataset == 'observer1mmtranshann':
            data_dir_temp = glob.glob(os.path.join(dir_, 'Observer_1mm_trans_hann', '*'))
            mat_file = io.loadmat(os.path.join(dir_, 'Observer_1mm_trans_hann', 'label.mat'))
            data_dir_temp = [d for d in data_dir_temp if os.path.isdir(d)]
        else:
            data_dir_temp = glob.glob(os.path.join(dir_, dataset, "*"))
            assert data_dir_ is not [], 'There are no such data name {0}'.format(dataname)
            data_dir_temp = [d for d in data_dir_temp if os.path.isdir(d)]
        # label = np.concatenate([label, mat_file['label']], axis=1)
        data_dir_ = data_dir_ + data_dir_temp
    numlabel = 2
    datalist1 = [[], [], []]
    datalist2 = [[], [], []]
    templist1 = []
    templist2 = []
    templist3 = []
    templist4 = []
    kindsnum = int(len(data_dir_)/len(dataname))
    for i in range(len(dataname)):  # 4
        templist1 = templist1 + glob.glob(os.path.join(data_dir_[0 + kindsnum * i], '*'))  # 80000
        templist2 = templist2 + glob.glob(os.path.join(data_dir_[1 + kindsnum * i], '*'))
        # templist3 = templist3 + glob.glob(os.path.join(data_dir_[1 + 4 * i], '*'))
        # templist4 = templist4 + glob.glob(os.path.join(data_dir_[1 + 4 * i], '*'))
    datanum = int(len(templist1) / len(dataname))  # 20000
    numval = int(np.floor(valrate * datanum))
    numtest = int(np.floor(testrate * datanum))
    numtrain = datanum - numval - numtest
    print("Dataset Num of train: {}, Num of Val : {}, Num of test : {}".format(numtrain, numval, numtest))
    for j in range(len(dataname)):  # 4
        for i in range(datanum):  # 20000
            # tempimg1 = readimage(templist1[i + datanum * j])
            # tempimg2 = readimage(templist2[i + datanum * j])
            # tempimg = [tempimg1, tempimg2]
            tempimg = [templist1[i + datanum * j], templist2[i + datanum * j]]#, templist3[i + datanum * j]] #, templist4[i + datanum * j]]
            if i >= numtrain + numval:
                # templabel1 = np.asarray([1, 0], dtype=np.int8)
                # templabel2 = np.asarray([0, 1], dtype=np.int8)
                datalist1[2] += [tempimg[0]]
                datalist2[2] += [tempimg[1]]
                # datalist2[2] += [tempimg[2]]
                # datalist2[2] += [tempimg[3]]
            elif i >= numtrain:
                # templabel1 = np.asarray([1, 0], dtype=np.int8)
                # templabel2 = np.asarray([0, 1], dtype=np.int8)
                datalist1[1] += [tempimg[0]]
                datalist2[1] += [tempimg[1]]
                # datalist2[1] += [tempimg[2]]
                # datalist2[1] += [tempimg[3]]
            else:
                # templabel1 = np.asarray([1, 0], dtype=np.int8)
                # templabel2 = np.asarray([0, 1], dtype=np.int8)
                datalist1[0] += [tempimg[0]]
                datalist2[0] += [tempimg[1]]
                # datalist2[0] += [tempimg[2]]
                # datalist2[0] += [tempimg[3]]

    binarylabellist = []
    binarylabellist.append(
        np.array([[1, 0] for _ in range(numtrain)] * len(dataname) + [[0, 1] for _ in range(numtrain)] * 1 * len(dataname)))
    binarylabellist.append(
        np.array([[1, 0] for _ in range(numval)] * len(dataname) + [[0, 1] for _ in range(numval)] * 1 * len(dataname)))
    binarylabellist.append(
        np.array([[1, 0] for _ in range(numtest)] * len(dataname) + [[0, 1] for _ in range(numtest)] * 1 * len(dataname)))

    class DataSets(object):
        pass

    datasets = DataSets()
    datasets.train = DataSet(
        np.concatenate((np.asarray(datalist1[0]), np.asarray(datalist2[0])), axis=0),
        binarylabellist[0]
    )
    if numval != 0:
        datasets.val = DataSet(
            np.concatenate((np.asarray(datalist1[1]), np.asarray(datalist2[1])), axis=0),
            binarylabellist[1]
        )

    if numtest != 0:
        datasets.test = DataSet(
            np.concatenate((np.asarray(datalist1[2]), np.asarray(datalist2[2])), axis=0),
            binarylabellist[2]
        )

    return datasets


def loadsampledata(dataname, sampledatanum=1000, dir_='D:\CTgit\Image', labeldice=1):
    data_dir_ = []
    label = np.zeros([4, 0])
    for dataset in dataname:
        if dataset == 'observer2mmtransramp':
            data_dir_temp = glob.glob(os.path.join(dir_, 'Observer_2mm_trans_ramp', '*'))
            mat_file = io.loadmat(os.path.join(dir_, 'Observer_2mm_trans_ramp', 'label.mat'))
            data_dir_temp = [d for d in data_dir_temp if os.path.isdir(d)]
        elif dataset == 'observer2mmtranshann':
            data_dir_temp = glob.glob(os.path.join(dir_, 'Observer_2mm_trans_hann', '*'))
            mat_file = io.loadmat(os.path.join(dir_, 'Observer_2mm_trans_hann', 'label.mat'))
            data_dir_temp = [d for d in data_dir_temp if os.path.isdir(d)]
        elif dataset == 'observer2mmlongiramp':
            data_dir_temp = glob.glob(os.path.join(dir_, 'Observer_2mm_longi_ramp', '*'))
            data_dir_temp = [d for d in data_dir_temp if os.path.isdir(d)]
        elif dataset == 'observer2mmlongihann':
            data_dir_temp = glob.glob(os.path.join(dir_, 'Observer_2mm_longi_hann', '*'))
            data_dir_temp = [d for d in data_dir_temp if os.path.isdir(d)]
        elif dataset == 'observer1mmtransramp':
            data_dir_temp = glob.glob(os.path.join(dir_, 'Observer_1mm_trans_ramp', '*'))
            mat_file = io.loadmat(os.path.join(dir_, 'Observer_1mm_trans_ramp', 'label.mat'))
            data_dir_temp = [d for d in data_dir_temp if os.path.isdir(d)]
        elif dataset == 'observer1mmtranshann':
            data_dir_temp = glob.glob(os.path.join(dir_, 'Observer_1mm_trans_hann', '*'))
            mat_file = io.loadmat(os.path.join(dir_, 'Observer_1mm_trans_hann', 'label.mat'))
            data_dir_temp = [d for d in data_dir_temp if os.path.isdir(d)]
        else:
            data_dir_temp = glob.glob(os.path.join(dir_, dataset, "*"))
            assert data_dir_ is not [], 'There are no such data name {0}'.format(dataname)
            data_dir_temp = [d for d in data_dir_temp if os.path.isdir(d)]
        # label = np.concatenate([label, mat_file['label']], axis=1)
        data_dir_ = data_dir_ + data_dir_temp
    datalist1 = []
    # binarylabellist = [np.array([], dtype=np.int8).reshape(0, numlabel) for _ in range(3)]
    print(data_dir_)

    templist1 = glob.glob(os.path.join(data_dir_[labeldice - 1], '*'))
    datanum = len(templist1)
    print("Dataset Num of sample : {}".format(sampledatanum))
    for i in range(sampledatanum):
        tempimg1 = templist1[datanum - 1 - i]
        datalist1 += [tempimg1]

    if labeldice == 1:
        templabel = [1, 0]
    else:
        templabel = [0, 1]

    binarylabellist = (np.array([templabel for _ in range(sampledatanum)]))

    datasets = DataSet(
        np.asarray(datalist1),
        binarylabellist
    )

    return datasets
