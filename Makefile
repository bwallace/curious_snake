.PHONY : all clean

all:
	cd learners/libsvm && $(MAKE)
clean:
	cd learners/libsvm && $(MAKE) clean