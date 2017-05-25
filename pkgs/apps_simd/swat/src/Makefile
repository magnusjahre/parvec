PREFIX=${PARSECDIR}/pkgs/apps_simd/swat/inst/${PARSECPLAT}


#CXX = g++

# Set this name equal to what you want the executable to be called
TARGET=opal_aligner


OBJS = test opal_aligner


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
	


all: $(OBJS)

opal_aligner: opal_wrapper.o ScoreMatrix.o opal_aligner.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) opal_wrapper.o ScoreMatrix.o opal_aligner.cpp $(LIBS) -o $(TARGET)

ScoreMatrix.o: ScoreMatrix.cpp ScoreMatrix.hpp
	$(CXX) $(CXXFLAGS) -c ScoreMatrix.cpp

test: opal_wrapper.o ScoreMatrix.o test.cpp
	$(CXX) $(CXXFLAGS) opal_wrapper.o ScoreMatrix.o test.cpp -o test

opal.o: opal_wrapper.cpp opal.h
	$(CXX) $(CXXFLAGS) -c opal_wrapper.cpp

clean:
	rm *.o test $(TARGET)

install:
	mkdir -p $(PREFIX)/bin
	cp -f $(TARGET) $(PREFIX)/bin/$(TARGET)
