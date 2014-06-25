picolisp:
	nvcc -arch=sm_30 --compiler-options '-fPIC' --shared -dc -g -G *.cu
#	nvcc -arch=sm_30 *.o -o pl

# Clean up
clean:
	rm -f *.o
