train_cuda:train.cu mnist.h mnist.cu fnn.h fnn.cu matrix.h matrix.cu
	nvcc -O2 train.cu mnist.cu fnn.cu matrix.cu -o train_cuda -lm