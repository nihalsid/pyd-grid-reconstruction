PROJECT_ROOT_DIR = .
PYTHON_LIBS_DIR = ${PYTHONENV_PATH}/lib/
PYTHON_LIBS = -lpython2.7
CUDA_LIBS = -lnppc -lnppim
PYTHON_INCLUDES_DIR = ${PYTHONENV_PATH}/include/python2.7/
NUMPY_INCLUDES_DIR = ${PYTHONENV_PATH}/lib/python2.7/site-packages/numpy/core/include
GAMMA_FILTER_PROJECT = ../gamma-spot-removal-gpu
OUTPUT_DIR = bin
OUTPUT_FILE = gridrecon.so
SOURCE_DIR = $(PROJECT_ROOT_DIR)/GridReconstructionPy
GAMMA_FILTER_SOURCE = $(GAMMA_FILTER_PROJECT)/GammaFilter
SOURCE_FILES = $(SOURCE_DIR)/gridrecon_py27.c $(SOURCE_DIR)/gridreconimpl.cpp $(GAMMA_FILTER_SOURCE)/cuda_ops.cu 
INCLUDE_DIRS = "$(GAMMA_FILTER_SOURCE),$(PYTHON_INCLUDES_DIR),$(NUMPY_INCLUDES_DIR)"
LIBRARY_DIRS = "$(PYTHON_LIBS_DIR)"
LIBRARIES = "$(PYTHON_LIBS) $(CUDA_LIBS)"
NVCC_FLAGS = "-std=c++11"
OUTPUT_PATH = $(OUTPUT_DIR)/$(OUTPUT_FILE)

all:
	mkdir -p $(OUTPUT_DIR)
	nvcc $(NVCC_FLAGS) -o $(OUTPUT_PATH) --compiler-options '-fPIC'  -shared $(SOURCE_FILES) -I $(INCLUDE_DIRS) -L $(LIBRARY_DIRS) $(LIBRARIES)

clean:
	rm -rf $(OUTPUT_DIR)
