# Makefile for nbody

PREFIX=${PARSECDIR}/pkgs/apps_simd/nbody/inst/${PARSECPLAT}


# Set this name equal to what you want the executable to be called
TARGET=gccnbody

# Set this name equal to what your program source code file is called
OBJS=nbody.o


ifdef version
	ifeq "$(version)" "pthreads"
		CXXFLAGS := $(CXXFLAGS) -DENABLE_THREADS -pthread -std=gnu99
	endif
	ifeq "$(version)" "tbb"
		CXXFLAGS := $(CXXFLAGS) -DTBB_VERSION -std=gnu99
	endif
endif

all: $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJS) $(LIBS) -o $(TARGET)

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f *.o $(TARGET)

install:
	mkdir -p $(PREFIX)/bin
	cp -f $(TARGET) $(PREFIX)/bin/$(TARGET)