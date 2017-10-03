import numpy as np
from guidedfilter import guidedfilter
from scipy.io import loadmat
from exrTools import exrread,exrwrite
import os



def processOneScene(rgbBsePath,featureBasePath,scene,outputPath,epsilon = 0.0001,windowSize = 11):
	rgbPath = rgbBsePath + scene[:-4] + '_MC_0016.exr'
	featurePath = featureBasePath + scene[:-4] + '.mat'
	p = np.log(1+exrread(rgbPath))
	print (featurePath)
	I = loadmat(featurePath)['FirstFeature']
	H = p.shape[0]
	W = p.shape[1]
	I = np.reshape(I,(H,W,18))
	I = I[...,5:]
	guidence = np.ones(shape=(H,W,5))
	guidence[..., 0] = threeConvertToOne(I[...,0:3])
	guidence[..., 1] = threeConvertToOne(I[...,3:6])
	guidence[..., 2] = threeConvertToOne(I[...,6:9])
	guidence[..., 3] = threeConvertToOne(I[...,9:12])
	guidence[..., 4] = I[...,12]
	# p = threeConvertToOne(p)
	name = ['_posi','_norm','_tex1','_tex2','_visi']
	
	for index in range(0,5):
		filtered = np.ones_like(p)
		filtered[..., 0] = guidedfilter(guidence[..., index], p[..., 0], windowSize, epsilon)
		filtered[..., 1] = guidedfilter(guidence[..., index], p[..., 1], windowSize, epsilon)
		filtered[..., 2] = guidedfilter(guidence[..., index], p[..., 2], windowSize, epsilon)
		exrwrite(filtered,
		         outputPath + scene[:-4] + name[index] + '.exr')


def threeConvertToOne(img):
	return (np.sqrt(img[...,0]*img[...,0] + img[...,1]*img[...,1] + img[...,2]*img[...,2]))


def main():
	rgbBsePath = '/workspace/data/16spp/exr/'
	featureBasePath = '/workspace/data/16spp/feature/'
	outputPath = '/workspace/data/16spp/guidedfilter/'
	#Gtpath = '/workspace/data/DatasetWithSecondFeature_aug3/GT/'
	scenes = os.listdir(featureBasePath)
	for scene in scenes:
		processOneScene(rgbBsePath = rgbBsePath,
		                featureBasePath = featureBasePath,
		                scene = scene,
		                outputPath=outputPath,
		                epsilon=0.0001,
		                windowSize=5
		                )

if __name__ == '__main__':
	main()
