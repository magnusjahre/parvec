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



# Enable SSE since we are running a scaled-down version of the config
CXXFLAGS := -std=c++11 -msse4.1 -DPARSEC_USE_SSE $(CXXFLAGS)

#DEBUG ?= 0
#ifeq ($(DEBUG), 1)
#	CXXFLAGS += -g -fno-omit-frame-pointer
#else
#	CXXFLAGS += -O3
#endif
	


default: $(OBJS)

opal_aligner: opal_scalar.o ScoreMatrix.o opal_aligner.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) opal_scalar.o ScoreMatrix.o opal_aligner.cpp $(LIBS) -o $(TARGET)

ScoreMatrix.o: ScoreMatrix.cpp ScoreMatrix.hpp
	$(CXX) $(CXXFLAGS) -c ScoreMatrix.cpp

test: opal_scalar.o ScoreMatrix.o test.cpp
	$(CXX) $(CXXFLAGS) opal_scalar.o ScoreMatrix.o test.cpp -o test

opal_scalar.o: opal_scalar.cpp opal.h
	$(CXX) $(CXXFLAGS) -c opal_scalar.cpp

clean:
	rm *.o test $(TARGET)

install:
	mkdir -p $(PREFIX)/bin
	cp -f $(TARGET) $(PREFIX)/bin/$(TARGET)
