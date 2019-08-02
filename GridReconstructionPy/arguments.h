#ifndef ARGUMENTS
#define ARGUMENTS

#include <Python.h>
#include <numpy/arrayobject.h>

struct Arguments {
	PyArrayObject* image_in;
	float thres3;
	float thres5;
	float thres7;
	float sig_log;
	PyArrayObject* image_out;
};

#endif
