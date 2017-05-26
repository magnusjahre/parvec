# Determine if application is compiling for vectorization or not
ifneq (,$(findstring PARSEC_USE_AVX,$(CXXFLAGS)))
	vectorized = true
endif
ifneq (,$(findstring PARSEC_USE_SSE,$(CXXFLAGS)))
	vectorized = true
endif

# Set this name equal to what your program source code file is called
ifdef vectorized
	path=vectorized
else
	path=scalar
endif



all:
	$(MAKE) -C $(path)

install:
	$(MAKE) -C $(path) install