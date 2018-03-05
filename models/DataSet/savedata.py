import scipy.misc
import os
import numpy as np
from PIL import Image


def congrateimg(targetimgs, size):
	h, w = targetimgs.shape[1], targetimgs.shape[2]
	img = np.zeros((h * size, w * size, targetimgs.shape[3]))
	for idx, image in enumerate(targetimgs):
		i = idx % size
		j = idx // size
		img[j*h:j*h+h, i*w:i*w+w, :] = image
	return img


def saveimg(targetimg, filename, isgrey=True, path='D:\\CTgit\\Image\\resultImg'):
	imgdim = targetimg.shape
	targetimg = np.reshape(targetimg, [imgdim[0], imgdim[1]])
	if len(imgdim) == 4-isgrey and imgdim[-1] != 1:
		imgnum = imgdim[0]
		for i in range(imgnum):
			scipy.misc.imsave(
				os.path.join(path, filename+'_'+str(i)+'.png'),
				targetimg[i].reshape([imgdim[1], imgdim[2], -1])
			)
	elif len(imgdim) == 3-isgrey or imgdim[-1] == 1:
		scipy.misc.toimage(targetimg, cmin=0, cmax=1).save(os.path.join(path, filename+'.png'))
		# scipy.misc.imsave(
		# 	os.path.join(path, filename+'.png'),
		# 	targetimg
		# )
	else:
		assert 0, 'Image dimension is strange'
