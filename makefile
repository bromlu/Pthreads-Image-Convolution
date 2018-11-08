convolution:
	gcc -pthread -Wall -Werror -o convolve convolveParallel.c lodepng.c

convolutionOld:
	gcc -Wall -Werror -o convolveOld convolve.c lodepng.c

clean:
	rm convolve 


