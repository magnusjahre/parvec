# Makefile for nqueens

PREFIX=${PARSECDIR}/pkgs/apps_simd/nqueens/inst/${PARSECPLAT}

# Set this name equal to what you want the executable to be called
TARGET=nqueens

# Determine if application is compiling for vectorization or not
ifneq (,$(findstring PARSEC_USE_AVX,$(CXXFLAGS)))
	vectorized = true
endif
ifneq (,$(findstring PARSEC_USE_SSE,$(CXXFLAGS)))
	vectorized = true
endif

# Set this name equal to what your program source code file is called
ifdef vectorized
	OBJS = nqueens-block-reexp-vect.o harness.o
else
    OBJS = nqueens-block-reexp-scalar.o harness.o
endif


ifdef version
	ifeq "$(version)" "pthreads"
		CXXFLAGS := $(CXXFLAGS) -DENABLE_THREADS -pthread
	endif
	ifeq "$(version)" "tbb"
		CXXFLAGS := $(CXXFLAGS) -DTBB_VERSION
	endif
endif


default : $(OBJS)
	$(CXX) $(LDFLAGS) -o $(TARGET) $^ $(LIBS)

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -I. -o $@ -c $<

harness.o: harness.cpp harness.h
	$(CXX) $(CXXFLAGS) -c harness.cpp

clean:
	rm -f *.o $(TARGET)

install:
	mkdir -p $(PREFIX)/bin
	cp -f $(TARGET) $(PREFIX)/bin/$(TARGET)
