#define PY_ARRAY_UNIQUE_SYMBOL gridrecon_ARRAY_API
#include <Python.h>
#include <numpy/arrayobject.h>


int createAndFillLogFilter(int N, float sigma, PyArrayObject* out_array);
int applyGammaFilter(PyArrayObject* image_in, float thres3, float thres5, float thres7, float sig_log, PyArrayObject* image_out);

/*  wrapped function */
static PyObject* log_filter(PyObject* self, PyObject* args)
{
	int N;
	float sigma;
	PyObject *out_array = NULL;
	PyArray_Descr *dtype = NULL;
	///*  parse the input, from python float to c double */
	if (!PyArg_ParseTuple(args, "if", &N, &sigma))
		return NULL;
	///* if the above function returns -1, an appropriate Python exception will
	//* have been set, and the function simply returns NULL
	//*/
	int nd = 2;
	int dims[2] = {N, N};
	printf("Sigma: %f\n", sigma);
	out_array = (PyArrayObject *) PyArray_FromDims(nd, dims, NPY_FLOAT);
	createAndFillLogFilter(N, (float)sigma, out_array);

	return PyArray_Return(out_array);
}

PyArrayObject *pymatrix(PyObject *objin) {
	return (PyArrayObject *)PyArray_ContiguousFromObject(objin, NPY_FLOAT, 2, 2);
}

/*  wrapped function */
static PyObject* gam_rem_adp_log(PyObject* self, PyObject* args) {
	PyArrayObject *image_in, *image_out;
	float thr3, thr5, thr7;
	float sig_log;
	/* Parse tuples separately since args will differ between C fcns */
	if (!PyArg_ParseTuple(args, "O!O!ffff", &PyArray_Type, &image_in, &PyArray_Type, &image_out, &thr3, &thr5, &thr7, &sig_log))
		return NULL;

	applyGammaFilter(image_in, thr3, thr5, thr7, sig_log, image_out);
	return Py_True;
}


/*  define functions in module */
static PyMethodDef GridReconMethods[] =
{
	{ "log_filter", log_filter, METH_VARARGS, "create the log filter" },
	{ "gam_rem_adp_log", gam_rem_adp_log , METH_VARARGS, "apply gamma filter"},
	{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef cGridReconModule =
{
	PyModuleDef_HEAD_INIT,
	"gridrecon", "Python module for grid reconstruction for FRMII",
	-1,
	GridReconMethods
};

PyMODINIT_FUNC
PyInit_gridrecon(void)
{
	import_array();
	return PyModule_Create(&cGridReconModule);
}
