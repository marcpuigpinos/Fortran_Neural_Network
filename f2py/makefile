build:
	f2py -c fnn.f90 -llibfnn.a -m fortran -I$(shell pwd)

run:
	python fnn.py

clean:
	rm *.so
