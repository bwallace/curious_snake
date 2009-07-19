# Prerequisites: python source, numpy
#
# On Ubuntu:
# sudo apt-get install python-dev
# sudo apt-get install python-numpy

.PHONY : all clean

all:
	cd learners/libsvm && $(MAKE)
clean:
	cd learners/libsvm && $(MAKE) clean