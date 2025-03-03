build:
	cp ../src/Fortran_Neural_Network.f90 .
	gfortran -c Fortran_Neural_Network.f90
	gfortran -c fnn.f90
	python -m numpy.f2py fnn.f90 -m fnn -h fnn.pyf
	python -m numpy.f2py -c --noarch --f90exec=gfortran --fcompiler=gfortran fnn.pyf fnn.o Fortran_Neural_Network.o -I$(shell pwd)

run:
	python fnn.py

clean:
	rm *.so Fortran_Neural_Network.f90 *.mod *.o *.pyf 
