import cv2
import numpy as np
from guidedfilter import guidedfilter
from scipy.io import loadmat
from exrTools import exrread,exrwrite
import os



def processOneScene(rgbBsePath,featureBasePath,scene,outputPath,spp = '4spp',epsilon = 0.0001,windowSize = 11):
	
	if spp == '4spp':
		rgbPath = rgbBsePath + scene[:-4] + '_MC_0004.exr'
	else if spp == '8spp':
		rgbPath = rgbBsePath + scene[:-4] + '_MC_0008.exr'
	else if spp == '16spp':
		rgbPath = rgbBsePath + scene[:-4] + '_MC_0016.exr'
	else if spp == '32spp':
		rgbPath = rgbBsePath + scene[:-4] + '_MC_0032.exr'
	else if spp == '64spp':
		rgbPath = rgbBsePath + scene[:-4] + '_MC_0064.exr'

	featurePath = featureBasePath + scene[:-4] + '.mat'
	p = exrread(rgbPath)
	print (featurePath)
	I = loadmat(featurePath)['FirstFeature']
	H = p.shape[0]
	W = p.shape[1]
	I = np.reshape(I,(H,W,21))
	I = I[...,5:]
	guidence = np.ones(shape=(H,W,9))

	guidence[..., 0] = I[...,13] #depth with project to camera normal
	guidence[..., 1] = I[..., 14] # norm with project to camera normal
	guidence[..., 2] =I[...,12] # visibility

	#tex1
	guidence[..., 3] = I[...,6]
	guidence[..., 4] = I[...,7]
	guidence[...,5] = I[...,8]
	# tex2
	guidence[..., 6] = I[..., 9]
	guidence[..., 7] = I[..., 10]
	guidence[..., 8] = I[..., 11]

	name = ['_depth','_normProject','_vis','_tex1','_tex2']
	
	for index in range(0,3):
		filtered = np.ones_like(p)
		filtered[..., 0] = guidedfilter(guidence[..., index], p[..., 0], windowSize, epsilon)
		filtered[..., 1] = guidedfilter(guidence[..., index], p[..., 1], windowSize, epsilon)
		filtered[..., 2] = guidedfilter(guidence[..., index], p[..., 2], windowSize, epsilon)
		exrwrite(filtered,
		         outputPath + scene[:-4] + name[index] + '.exr')

	filtered = np.ones_like(p)
	filtered[..., 0] = guidedfilter(guidence[..., 3], p[..., 0], windowSize, epsilon)
	filtered[..., 1] = guidedfilter(guidence[..., 4], p[..., 1], windowSize, epsilon)
	filtered[..., 2] = guidedfilter(guidence[..., 5], p[..., 2], windowSize, epsilon)
	exrwrite(filtered,
	         outputPath + scene[:-4] + name[3] + '.exr')

	filtered = np.ones_like(p)
	filtered[..., 0] = guidedfilter(guidence[..., 6], p[..., 0], windowSize, epsilon)
	filtered[..., 1] = guidedfilter(guidence[..., 7], p[..., 1], windowSize, epsilon)
	filtered[..., 2] = guidedfilter(guidence[..., 8], p[..., 2], windowSize, epsilon)
	exrwrite(filtered,
	     outputPath + scene[:-4] + name[4] + '.exr')



def main():
	for SPP in ['4spp','8spp','16spp','32spp','64spp']:
		rgbBsePath = '/workspace/data/'+SPP+'/exr/'
		featureBasePath = '/workspace/data/'+SPP+'/feature/'
		outputPath = '/workspace/data/'+SPP+'/guidedfilter/'
		scenes = os.listdir(featureBasePath)
		for scene in scenes:
			processOneScene(rgbBsePath = rgbBsePath,
			                featureBasePath = featureBasePath,
			                scene = scene,
			                outputPath=outputPath,
			                spp = SPP,
			                epsilon=0.0001,
			                windowSize=5
			                )

if __name__ == '__main__':
	main()