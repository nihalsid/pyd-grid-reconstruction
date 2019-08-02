import sys
sys.path.insert(0,'/home/nihalsid/Documents/FRM-II/git/pyd-grid-reconstruction/bin')
import gridrecon
import unittest
import os
import gridrecon
from gam_rem_adp_log import gam_rem_adp_log
import numpy as np
import pyfits
import matplotlib.pyplot as plt
import time


def compare_arrays(array1, array2):
	isclose = np.isclose(array1, array2, rtol=1e-2, atol=1e-5)
	inner_isclose = np.isclose(array1[50:-50,50:-50], array2[50:-50,50:-50], rtol=1e-2, atol=1e-5)
	return np.sum(isclose == True) * 1.0 / (array1.shape[0] * array1.shape[1]), np.sum(inner_isclose == True) * 1.0 / (inner_isclose.shape[0] * inner_isclose.shape[1])


def test_gam_rem_adp_log(filelist):
	xmin=200 # As measured in ImageJ (hopefully)
	xmax=1000
	ymin=45
	ymax=2000	
	
	reads = []
	writes = []
	for f in filelist:
		reads.append(np.ascontiguousarray(pyfits.open(f)[0].data[ymin:ymax,xmin:xmax], dtype=np.float32))
		writes.append(np.ascontiguousarray(np.zeros(reads[-1].shape, dtype=np.float32), dtype=np.float32))
	
	t1 = time.time()
	gridrecon.gam_rem_adp_log(reads, writes, 50, 100, 200, 0.8, 4)
	print('C++: ', time.time() - t1)
	python_outputs = []
	
	t1 = time.time()
	for r in reads:
		python_outputs.append(gam_rem_adp_log(r,50,100,200,0.8))

	print('Python: ', time.time() - t1)
	
	for i in range(len(reads)):
		close, inner_close = compare_arrays(writes[i], python_outputs[i])
		assert inner_close >= (1 - 1e-5), "Inner closeness value = %.4f" % inner_close
		assert close >= 0.99, "Closeness value = %.4f" % close
		#print("Inner closeness value = %.4f [~1]" % inner_close)
		#print("Closeness value = %.4f [~0.99]" % close)

if __name__=='__main__':
	test_images_dir = r'../../fits'
	test_images = [os.path.join(test_images_dir, x) for x in os.listdir(test_images_dir) if x.endswith('.fits')]
	test_images = test_images[:8]
	test_gam_rem_adp_log(test_images)

