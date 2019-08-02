#define PY_ARRAY_UNIQUE_SYMBOL gridrecon_ARRAY_API
#include <Python.h>
#include <numpy/arrayobject.h>
#include <pthread.h>
#include "arguments.h"
#define MAX_NUM_THREADS 128
#define MIN(a, b) ((a) < (b) ? (a) : (b))

int createAndFillLogFilter(int N, float sigma, PyArrayObject* out_array);
void* applyGammaFilter(void* args);

/*  wrapped function */
static PyObject* log_filter(PyObject* self, PyObject* _args)
{
	int N;
	float sigma;
	PyObject *out_array = NULL;
	PyArray_Descr *dtype = NULL;
	///*  parse the input, from python float to c double */
	if (!PyArg_ParseTuple(_args, "if", &N, &sigma))
		return NULL;
	///* if the above function returns -1, an appropriate Python exception will
	//* have been set, and the function simply returns NULL
	//*/
	int nd = 2;
	int dims[2] = {N, N};
	out_array = (PyArrayObject *) PyArray_FromDims(nd, dims, NPY_FLOAT);
	createAndFillLogFilter(N, (float)sigma, out_array);

	return PyArray_Return(out_array);
}

PyArrayObject *pymatrix(PyObject *objin) {
	return (PyArrayObject *)PyArray_ContiguousFromObject(objin, NPY_FLOAT, 2, 2);
}

/*  wrapped function */
static PyObject* gam_rem_adp_log(PyObject* self, PyObject* args) {
	//PyArrayObject *image_in, *image_out;
	PyObject* p_list_in, *p_list_out;
	float thr3, thr5, thr7;
	float sig_log;
	int num_threads;
	/* Parse tuples separately since args will differ between C fcns */
	if (!PyArg_ParseTuple(args, "O!O!ffffi", &PyList_Type, &p_list_in, &PyList_Type, &p_list_out, &thr3, &thr5, &thr7, &sig_log, &num_threads))
		return NULL;
	pthread_t thread_ids[MAX_NUM_THREADS];
	int total_images = PyList_Size(p_list_in);
	int current_image_index = 0;
	while(current_image_index < total_images){
		int num_required_threads = MIN(total_images - current_image_index, num_threads); 
		for (int i = 0; i < num_required_threads; i++) {
			PyArrayObject* image_in = PyList_GetItem(p_list_in, current_image_index);
			PyArrayObject* image_out = PyList_GetItem(p_list_out, current_image_index);
			struct Arguments* _args = malloc(sizeof(struct Arguments));
			_args->image_in = image_in;
			_args->thres3 = thr3;
			_args->thres5 = thr5;
			_args->thres7 = thr7;
			_args->sig_log = sig_log;
			_args->image_out = image_out;
			assert(pthread_create(&thread_ids[i], NULL, applyGammaFilter, _args) == 0);
			current_image_index++;
		}
		for (int i = 0; i < num_required_threads; i++) {
			assert(pthread_join(thread_ids[i], NULL) == 0);
		}
	}
	Py_RETURN_TRUE;
}


/*  define functions in module */
static PyMethodDef GridReconMethods[] =
{
	{ "log_filter", log_filter, METH_VARARGS, "create the log filter" },
	{ "gam_rem_adp_log", gam_rem_adp_log , METH_VARARGS, "apply gamma filter"},
	{ NULL, NULL, 0, NULL }
};


PyMODINIT_FUNC
initgridrecon(void)
{
	(void) Py_InitModule("gridrecon", GridReconMethods);
	import_array();
}
