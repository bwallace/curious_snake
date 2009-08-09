CXX? = g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 1

.PHONY : python all clean

all: svm-train svm-predict svm-scale python

lib: svm.o
	$(CXX) -shared svm.o -o libsvm.so.$(SHVER)

svm-predict: svm-predict.c svm.o
	$(CXX) $(CFLAGS) svm-predict.c svm.o -o svm-predict -lm
svm-train: svm-train.c svm.o
	$(CXX) $(CFLAGS) svm-train.c svm.o -o svm-train -lm
svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) svm-scale.c -o svm-scale
svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp
clean:
	rm -f *~ svm.o svm-train svm-predict svm-scale
	cd python && $(MAKE) clean
python:
# not necessary
#	cd python && $(MAKE)
	cd python && python setup.py build build_ext --inplace