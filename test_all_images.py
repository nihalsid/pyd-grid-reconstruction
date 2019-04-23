import sys
sys.path.insert(0,r'C:\Users\Yawar\Documents\FRM-II\c++\GridReconstructionPy\x64\Release')
import gridrecon
import unittest
import os
import gridrecon
from gam_rem_adp_log import gam_rem_adp_log
import numpy as np
import pyfits
import matplotlib.pyplot as plt


def compare_arrays(array1, array2):
	isclose = np.isclose(array1, array2, rtol=1e-2, atol=1e-5)
	inner_isclose = np.isclose(array1[50:-50,50:-50], array2[50:-50,50:-50], rtol=1e-2, atol=1e-5)
	return np.sum(isclose == True) / (array1.shape[0] * array1.shape[1]), np.sum(inner_isclose == True) / (inner_isclose.shape[0] * inner_isclose.shape[1])


class ParameterizedTestCase(unittest.TestCase):
	'''
	Inspired from https://eli.thegreenplace.net/2011/08/02/python-unit-testing-parametrized-test-cases
	'''
	def __init__(self, methodName='runTest', param=None):
		super(ParameterizedTestCase, self).__init__(methodName)
		self.param = param

	@staticmethod
	def parameterize(testcase_klass, param=None):
		testloader = unittest.TestLoader()
		testnames = testloader.getTestCaseNames(testcase_klass)
		suite = unittest.TestSuite()
		for name in testnames:
			suite.addTest(testcase_klass(name, param=param))
		return suite


class TestCudaVsPython(ParameterizedTestCase):

	def test_gam_rem_adp_log(self):
		xmin=200 # As measured in ImageJ (hopefully)
		xmax=1000
		ymin=45
		ymax=2000	
		img =  np.ascontiguousarray(pyfits.open(self.param)[0].data[ymin:ymax,xmin:xmax], dtype=np.float32)
		cuda_output = np.ascontiguousarray(np.zeros(img.shape, dtype=np.float32), dtype=np.float32)
		python_output = gam_rem_adp_log(img,50,100,200,0.8)
		gridrecon.gam_rem_adp_log(img, cuda_output, 50,100,200,0.8)
		close, inner_close = compare_arrays(cuda_output, python_output)
		assert inner_close >= (1 - 1e-5), "Inner closeness value = %.4f" % inner_close
		assert close >= 0.99, "Closeness value = %.4f" % close


	def __str__(self):
		return "%s (%s)" % (self._testMethodName, os.path.basename(self.param))


if __name__=='__main__':

	test_images_dir = r'C:\Users\Yawar\Documents\FRM-II\fits'
	test_images = [x for x in os.listdir(test_images_dir) if x.endswith('.fits')]

	suite = unittest.TestSuite()
	for test_image in test_images:
		path = os.path.join(test_images_dir, test_image)
		suite.addTest(ParameterizedTestCase.parameterize(TestCudaVsPython, param=path))

	unittest.TextTestRunner(verbosity=2).run(suite)
