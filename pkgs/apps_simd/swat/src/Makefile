PREFIX=${PARSECDIR}/pkgs/apps_simd/swat/inst/${PARSECPLAT}


#CXX = g++

# Set this name equal to what you want the executable to be called
TARGET=opal_aligner


#OBJS = test opal_aligner


ifdef version
	ifeq "$(version)" "pthreads"
		CXXFLAGS := $(CXXFLAGS) -DENABLE_THREADS -pthread
	endif
	ifeq "$(version)" "tbb"
		CXXFLAGS := $(CXXFLAGS) -DTBB_VERSION
	endif
endif



# -march=native will compile for correct simd
CXXFLAGS := -std=c++11 $(CXXFLAGS)

#DEBUG ?= 0
#ifeq ($(DEBUG), 1)
#	CXXFLAGS += -g -fno-omit-frame-pointer
#else
#	CXXFLAGS += -O3
#endif
	


all: test $(TARGET)

opal_aligner: opal.o ScoreMatrix.o opal_aligner.cpp
	$(CXX) $(CXXFLAGS) opal.o ScoreMatrix.o opal_aligner.cpp -o opal_aligner

ScoreMatrix.o: ScoreMatrix.cpp ScoreMatrix.hpp
	$(CXX) $(CXXFLAGS) -c ScoreMatrix.cpp

test: opal.o ScoreMatrix.o test.cpp
	$(CXX) $(CXXFLAGS) opal.o ScoreMatrix.o test.cpp -o test

opal.o: opal.cpp opal.h
	$(CXX) $(CXXFLAGS) -c opal.cpp

clean:
	rm *.o test opal_aligner

install:
	mkdir -p $(PREFIX)/bin
	cp -f $(TARGET) $(PREFIX)/bin/$(TARGET)
