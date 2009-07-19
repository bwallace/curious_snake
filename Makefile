# Prerequisites: python source, numpy, pythonbio
#
# On Ubuntu:
# sudo apt-get install python-dev
# sudo apt-get install python-numpy
# wget http://pypi.python.org/packages/2.6/s/setuptools/setuptools-0.6c9-py2.6.egg
# sudo sh setuptools-0.6c9-py2.6.egg
# sudo easy_install -f http://biopython.org/DIST/ biopython


.PHONY : all clean

all:
	cd learners/libsvm && $(MAKE)
clean:
	cd learners/libsvm && $(MAKE) clean