INCLUDE_PATH = ../include
SOURCE_PATH = src/search.cpp
BUILD_PATH = build/search

NVCC_FLAGS = -Xcompiler -fopenmp -lgomp -gencode arch=compute_80,code=sm_80 -O3 --include-path $(INCLUDE_PATH) -x cu -g

NVCC = nvcc

.ONESHELL:

all: $(BUILD_PATH)

$(BUILD_PATH):
	module load gcc
	module load cuda
	$(NVCC) $(NVCC_FLAGS) -o $@ $(SOURCE_PATH)
	cuobjdump -ptx $@ > $@.ptx
	cuobjdump -sass $@ > $@.sass

clean:
	rm -f $(BUILD_PATH) $(BUILD_PATH).ptx $(BUILD_PATH).sass

