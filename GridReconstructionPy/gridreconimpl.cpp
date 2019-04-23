#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL gridrecon_ARRAY_API

#include <numpy/arrayobject.h>

#include "cuda_ops.h"
#include <cuda_runtime.h>
#include <npp.h>

#include <memory>

#define ERROR_CHECK \
	gpuErrchk(cudaPeekAtLastError()); \
	gpuErrchk(cudaDeviceSynchronize());
# define M_PI 3.14159265358979323846
#define FLAT_ACCESS(R, C, W) ((R) * (W) + (C))


std::shared_ptr<float> createLogFilter(int N, float sigma) {
	if (N % 2 == 0) {
		N++;
	}
	int N2 = N / 2;
	std::shared_ptr<float> G(new float[N * N], std::default_delete<float[]>());
	std::shared_ptr<float> log(new float[N * N], std::default_delete<float[]>());

	float sum_G = 0.f;
	for (int i = -N2; i <= N2; i++) {
		for (int j = -N2; j <= N2; j++) {
			G.get()[FLAT_ACCESS(i + N2, j + N2, N)] = expf(-(i*i + j * j) / (2.0 * sigma * sigma));
			sum_G += G.get()[FLAT_ACCESS(i + N2, j + N2, N)];
		}
	}

	for (int i = -N2; i <= N2; i++) {
		for (int j = -N2; j <= N2; j++) {
			log.get()[FLAT_ACCESS(i + N2, j + N2, N)] = -1 * (i * i + j * j - 2 * sigma * sigma) * G.get()[FLAT_ACCESS(i + N2, j + N2, N)] / (2 * M_PI * powf(sigma, 6) * sum_G);
		}
	}

	return log;
}

extern "C" int createAndFillLogFilter(int N, float sigma, PyArrayObject* out_array) {
	std::shared_ptr<float> log_filter = createLogFilter(N, sigma);
	float* ptr_out_array = (float*) out_array->data;
	for (int i = 0; i < N * N; i++) {
		ptr_out_array[i] = log_filter.get()[i];
	}
	for (int i = 0; i < N; i++)
		ptr_out_array[i] = 0;
	return 0;
}


extern "C" float* createBoxFilter(int N) {
	float* box = (float*)malloc(sizeof(float) * N * N);
	for (int i = 0; i < N * N; i++) {
		box[i] = 1.f / (N * N);
	}
	return box;
}

extern "C" float **ptrvector(long n) {
	float **v;
	v = (float**)malloc((size_t)(n * sizeof(float)));
	if (!v) {
		printf("In **ptrvector. Allocation of memory for double array failed.");
		exit(0);
	}
	return v;
}

extern "C" float **pymatrix_to_Carrayptrs(PyArrayObject *arrayin) {
	float **c, *a;
	int i, n, m;
	n = arrayin->dimensions[0];
	m = arrayin->dimensions[1];
	c = ptrvector(n);
	a = (float *)arrayin->data;  /* pointer to arrayin data as double */
	
	for (i = 0; i<n; i++) {
		c[i] = a + i * m;
	}
	
	return c;
}

extern "C" void free_Carrayptrs(float **v) {
	free((char*)v);
}

extern "C" int applyGammaFilter(PyArrayObject* image_in, float thres3, float thres5, float thres7, float sig_log, PyArrayObject* image_out) {
	int height = image_in->dimensions[0];
	int width = image_in->dimensions[1];
	
	float* h_image_in = (float*)image_in->data;
	float* h_image_out = (float*)image_out->data;
	
	float* d_log_filter = nullptr;
	float* d_box_filter_normalized = nullptr;
	float* d_image_in = nullptr;
	float* d_image_log = nullptr;
	float* d_image_log_m3 = nullptr;
	float* d_image_buffer_0 = nullptr;
	float* d_image_buffer_1 = nullptr;
	float* d_image_adp = nullptr;

	bool* d_image_thres3 = nullptr;
	bool* d_image_thres5 = nullptr;
	bool* d_image_thres7 = nullptr;
	bool* d_image_single7 = nullptr;
	bool* d_bool_buffer = nullptr;
	bool* d_true_mask = nullptr;

	float* h_single_7 = createBoxFilter(3);

	std::shared_ptr<float> log_filter = createLogFilter(9, sig_log);

	initialize_gpu_buffer<float>(log_filter.get(), &d_log_filter, 9, 9);
	initialize_gpu_buffers<float, float>(h_image_in, &d_image_in, &d_image_log, width, height);
	create_gpu_buffer<float>(&d_image_log_m3, width, height);
	create_gpu_buffer<float>(&d_image_buffer_0, width, height);
	create_gpu_buffer<float>(&d_image_buffer_1, width, height);
	create_gpu_buffer<float>(&d_image_adp, width, height);
	create_gpu_buffer<float>(&d_box_filter_normalized, 3, 3);
	create_gpu_buffer<bool>(&d_image_thres3, width, height);
	create_gpu_buffer<bool>(&d_image_thres5, width, height);
	create_gpu_buffer<bool>(&d_image_thres7, width, height);
	create_gpu_buffer<bool>(&d_image_single7, width, height);
	create_gpu_buffer<bool>(&d_bool_buffer, width, height);
	create_gpu_buffer<bool>(&d_true_mask, width, height);

	set_to_true(d_true_mask, width, height);
	move_host_data_to_gpu(d_box_filter_normalized, h_single_7, 3, 3);

	convolve(d_image_in, d_image_log, log_filter.get(), width, height, 9);
	ERROR_CHECK;
	median_filter(d_image_log, d_image_log_m3, width, height, 3, d_true_mask);
	ERROR_CHECK;
	greater_than(d_image_log, d_image_log_m3, d_image_thres3, width, height, thres3);
	greater_than(d_image_log, d_image_log_m3, d_image_thres5, width, height, thres5);
	bool thres7_nonzero = greater_than(d_image_log, d_image_log_m3, d_image_thres7, width, height, thres7);
	ERROR_CHECK;
	if (thres7_nonzero) {
		multiply_constant(d_image_thres7, d_image_buffer_0, width, height, 255.f);
		ERROR_CHECK;
		convolve(d_image_buffer_0, d_image_buffer_1, d_box_filter_normalized, width, height, 3);
		ERROR_CHECK;
		less_than_constant(d_image_buffer_1, d_image_single7, width, height, 30.f);
		ERROR_CHECK;
		logical_operation(d_image_single7, d_image_thres7, d_image_single7, width, height, BooleanOperation::AND);
		ERROR_CHECK;
		logical_operation(d_image_thres7, d_image_single7, d_image_thres7, width, height, BooleanOperation::XOR);
		ERROR_CHECK;
		multiply_constant(d_image_thres7, d_image_buffer_0, width, height, 255.f);
		ERROR_CHECK;
		nppiDilate3x3_32f_C1R(d_image_buffer_0, width * sizeof(int), d_image_buffer_1, width * sizeof(int), { width, height });
		ERROR_CHECK;
		greater_than_constant(d_image_buffer_1, d_bool_buffer, width, height, 0.);
		ERROR_CHECK;
		logical_operation(d_bool_buffer, d_image_single7, d_image_thres7, width, height, BooleanOperation::OR);
		ERROR_CHECK;
	}

	logical_operation(d_image_thres5, d_image_thres7, d_bool_buffer, width, height, BooleanOperation::OR);
	ERROR_CHECK;
	logical_operation(d_bool_buffer, d_image_thres7, d_image_thres5, width, height, BooleanOperation::XOR);
	ERROR_CHECK;
	logical_operation(d_image_thres3, d_image_thres5, d_image_thres3, width, height, BooleanOperation::XOR);
	ERROR_CHECK;

	clone_buffer(d_image_in, d_image_adp, width, height);
	median_filter(d_image_in, d_image_buffer_0, width, height, 3, d_true_mask);
	ERROR_CHECK;
	masked_assign(d_image_adp, d_image_buffer_0, width, height, d_image_thres3);
	ERROR_CHECK;
	median_filter(d_image_in, d_image_adp, width, height, 5, d_image_thres5);
	ERROR_CHECK;
	median_filter(d_image_in, d_image_adp, width, height, 7, d_image_thres7);
	ERROR_CHECK;

	//multiply_constant(d_image_thres3, d_image_buffer_0, width, height, 1.f);

	move_gpu_data_to_host<float>(h_image_out, d_image_adp, width, height);
	 
	free_gpu_buffers<float, float>(d_image_in, d_image_log);
	free_gpu_buffer<float>(d_image_log_m3);
	free_gpu_buffer<float>(d_image_buffer_0);
	free_gpu_buffer<float>(d_image_buffer_1);
	free_gpu_buffer<float>(d_image_adp);
	free_gpu_buffer<float>(d_log_filter);
	free_gpu_buffer<float>(d_box_filter_normalized);
	free_gpu_buffer<bool>(d_image_thres3);
	free_gpu_buffer<bool>(d_image_thres5);
	free_gpu_buffer<bool>(d_image_thres7);
	free_gpu_buffer<bool>(d_image_single7);
	free_gpu_buffer<bool>(d_bool_buffer);
	free_gpu_buffer<bool>(d_true_mask);
	
	return 0;
}
