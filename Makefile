.PHONY : all clean

all:
	cd learners/svm_learners/libsvm && $(MAKE)
clean:
	cd learners/svm_learners/libsvm && $(MAKE) clean