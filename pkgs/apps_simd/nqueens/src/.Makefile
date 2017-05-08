# Makefile for nqueens

PREFIX=${PARSECDIR}/pkgs/apps_simd/nqueens/inst/${PARSECPLAT}

# Set this name equal to what you want the executable to be called
TARGET=nqueens-block-reexp-sse

# Set this name equal to what your program source code file is called
OBJS = nqueens-block-reexp-sse.o harness.o

ifdef version
	ifeq "$(version)" "pthreads"
		CXXFLAGS := $(CXXFLAGS) -DENABLE_THREADS -pthread
	endif
	ifeq "$(version)" "tbb"
		CXXFLAGS := $(CXXFLAGS) -DTBB_VERSION
	endif
endif



#Object:
#g++ -O2 -funroll-loops -Icommon -Iharness -Iblock -I. -o nqueens-block-reexp-sse.o -c nqueens-block-sreexp-sse.cpp

#Exxecutable:
#g++ -o nqueens-block-reexp-sse.x nqueens-block-reexp-sse.o harness.o -lpthread -lm



#DEPS = harness.cpp harness.h

#%.o : %.cpp $(OBJS)
#	$(CXX) $(CXXFLAGS) -o $(TARGET).o -c $<

#all : $(OBJS)
#	$(CXX) -o $(TARGET).x $^ $(CXXFLAGS)

.SUFFIXES: .cpp .x

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -I. -o $@ -c $<

harness.o: harness.cpp harness.h

nqueens-block-reexp-sse.x : $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f *.o $(TARGET)

install:
	mkdir -p $(PREFIX)/bin
	cp -f $(TARGET) $(PREFIX)/bin/$(TARGET)