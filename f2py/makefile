.PHONY: build
build:
	cp ../src/Fortran_Neural_Network.f90 .
	python -m numpy.f2py -c Fortran_Neural_Network.f90 -m fai

.PHONY: run
run:
	python fnn.py

.PHONY: clean
clean:
	rm -rf *.so Fortran_Neural_Network.f90 *.mod *.o *.pyf __pycache__
