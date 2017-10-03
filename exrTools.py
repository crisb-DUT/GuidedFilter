# -*- coding: utf-8 -*-
# @Author: cris
# @Date:   2017-05-11 13:43:11
# @Last Modified by:   cris
# @Last Modified time: 2017-09-28 15:46:26
import scipy.io as scio
import OpenEXR
import Imath
import numpy as np


def exrread(fileName):
	exrFile = OpenEXR.InputFile(fileName)
	header = exrFile.header()
	dw = header['dataWindow']
	pt = Imath.PixelType(Imath.PixelType.FLOAT)
	size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
	cc_r = np.fromstring(exrFile.channel('R', pt), dtype=np.float32)
	cc_g = np.fromstring(exrFile.channel('G', pt), dtype=np.float32)
	cc_b = np.fromstring(exrFile.channel('B', pt), dtype=np.float32)
	cc_r.shape = cc_g.shape = cc_b.shape = (size[1], size[0])
	cc = np.dstack((cc_r, cc_g, cc_b))
	return cc


def exrwrite(data,fileName):
	pixelsR = data[:,:,0].astype(np.float16).tostring()
	pixelsG = data[:,:,1].astype(np.float16).tostring()
	pixelsB = data[:,:,2].astype(np.float16).tostring()
	HEADER = OpenEXR.Header(data[:,:,0].shape[1],data[:,:,0].shape[0])
	half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
	HEADER['channels'] = dict([(c, half_chan) for c in "RGB"])
	exr = OpenEXR.OutputFile(fileName, HEADER)
	exr.writePixels({'R': pixelsR, 'G': pixelsG, 'B': pixelsB})
	exr.close()

