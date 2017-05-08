CC = g++
#CC = icc
#override CFLAGS += "-DBLAH"

ifeq ($(NOVEC),1)
	override CFLAGS += -no-vec 
endif
ifeq ($(NOINLINE), 1)
	override CFLAGS += -fno-inline-functions
endif 
LIBS = -lpthread -lm 

ifeq ($(CC),icc)
	override CFLAGS += -no-multibyte-chars -DICC
endif

ifeq ($(DEBUG),1)
  override CFLAGS += -DTRACK_TRAVERSALS
  ifeq ($(CC),icc)		
    override CFLAGS += -g -fno-inline-functions  
  else
	override CFLAGS += -g3 
  endif
else
  override CFLAGS += -O3 -DNDEBUG 
  ifeq ($(CC),icc)
    override CFLAGS += -ip -unroll  
  else
	override CFLAGS += -funroll-loops 
  endif
endif

ifeq ($(PROFILE),1)
	override CFLAGS += -DBLOCK_PROFILE
endif
ifeq ($(PROFILE),2)
	override CFLAGS += -DTREE_PROFILE
endif
ifeq ($(PROFILE),3)
	override CFLAGS += -DPROFILE
endif
ifeq ($(PROFILE),4)
	override CFLAGS += -DPARALLELISM_PROFILE
endif

ifeq ($(TRACK),1)
	override CFLAGS += -DTRACK_TRAVERSALS
endif

ifeq ($(SIMD_ALL),1)
	override CFLAGS += -DSIMD_ALL
endif

ifeq ($(SIMD_NONE),1)
	override CFLAGS += -DSIMD_NONE
endif

ifeq ($(SIMD_OPPORTUNITY),1)
	override CFLAGS += -DSIMD_OPPORTUNITY
endif

ifeq ($(NO_ELISION),1)
	override CFLAGS += -DNO_ELISION
endif

ifeq ($(BLOCK_TOP),1)
	override CFLAGS += -DBLOCK_TOP
endif

ifeq ($(TRACK_TRAVERSALS),1)
  override CFLAGS += -DTRACK_TRAVERSALS
endif

# for auto vectorizing with ICC
ifeq ($(AUTO_VECTORIZE),1)
  ifneq ($(CC),icc)
    $(error Auto vectorization only works for ICC.)
  endif
  override CFLAGS += -xAVX  -fno-alias -vec-report2 -DAUTOVEC
endif

# For saving intermediates
ifeq ($(KEEP),1)
  override CFLAGS += -save-temps
	ifeq ($(CC),icc)
    override CFLAGS += -masm=intel -use-asm 
  endif
endif

# for HeatRay
ifeq ($(ONE_PHASE),1)
	override CFLAGS += -DONE_PHASE
endif
ifeq ($(RAY_TRACE),1)
	override CFLAGS += -DRAY_TRACE
endif
ifeq ($(FULL_RENDER),1)
	override CFLAGS += -DFULL_RENDER
endif

ifeq ($(PAPI_FINE),1)
	override CFLAGS += -DPAPI_FINE
	PAPI = 1
endif

ifeq ($(PAPI),1)
	override CFLAGS += -DPAPI -I$(PAPI_PATH)
	HARNESS_OBJ += papiprofiler.o
	LIBS += -lpapi -L$(PAPI_PATH)
endif

HOST_CFLAGS  = -O3 -DNDEBUG  -unroll   -DSSE41 -D__SSE4_1 -msse4.1
HOST_LDFLAGS = 
MIC_CFLAGS   = -O3 -funroll-loops -mmic
MIC_LDFLAGS  = -mmic
