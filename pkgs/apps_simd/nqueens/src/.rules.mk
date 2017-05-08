
plain: \
nqueens-base.x
#fib-base.x                       \
#binomial-base.x                  \
#graphcol-base.x                  \
#knapsack-base.x                  \
#parentheses-base.x               \
#minmax-base.x                    \
#uts-base.x                       \
#fib-block-noreexp.x              \
#fib-block-reexp.x                \
#binomial-block-noreexp.x         \
#binomial-block-reexp.x           \
#nqueens-block-noreexp.x          \
#nqueens-block-reexp.x            \
#graphcol-block-noreexp.x         \
#graphcol-block-reexp.x           \
#knapsack-block-noreexp.x         \
#knapsack-block-reexp.x           \
#parentheses-block-noreexp.x      \
#parentheses-block-reexp.x        \
#minmax-block-noreexp.x           \
#minmax-block-reexp.x             \
#uts-block-noreexp.x              \
#uts-block-reexp.x

sse: \
nqueens-block-noreexp-sse.x      \
nqueens-block-reexp-sse.x
#fib-block-noreexp-sse.x          \
#fib-block-reexp-sse.x            \
#binomial-block-noreexp-sse.x     \
#binomial-block-reexp-sse.x       \
#graphcol-block-noreexp-sse.x     \
#graphcol-block-reexp-sse.x       \
#knapsack-block-noreexp-sse.x     \
#knapsack-block-reexp-sse.x       \
#parentheses-block-noreexp-sse.x  \
#parentheses-block-reexp-sse.x    \
#minmax-block-noreexp-sse.x       \
#minmax-block-reexp-sse.x         \
#uts-block-noreexp-sse.x          \
#uts-block-reexp-sse.x

avx: \
nqueens-block-noreexp-mic.x      \
nqueens-block-reexp-mic.x
#fib-block-noreexp-mic.x          \
#fib-block-reexp-mic.x            \
#binomial-block-noreexp-mic.x     \
#binomial-block-reexp-mic.x       \
#graphcol-block-noreexp-mic.x     \
#graphcol-block-reexp-mic.x       \
#knapsack-block-noreexp-mic.x     \
#knapsack-block-reexp-mic.x       \
#parentheses-block-noreexp-mic.x  \
#parentheses-block-reexp-mic.x    \
#minmax-block-noreexp-mic.x       \
#minmax-block-reexp-mic.x         \
#uts-block-noreexp-mic.x          \
#uts-block-reexp-mic.x

harness.o: harness.cpp harness.h

#knapsack
#knapsack-base.x: knapsack-base.o getoptions.o 

#knapsack-block-noreexp.x: knapsack-block-noreexp.o getoptions.o 

#knapsack-block-reexp.x: knapsack-block-reexp.o getoptions.o

#knapsack-block-noreexp-sse.x: knapsack-block-noreexp-sse.o getoptions.o

#knapsack-block-reexp-sse.x: knapsack-block-reexp-sse.o getoptions.o

#knapsack-block-noreexp-mic.x: knapsack-block-noreexp-mic.o getoptions.o

#knapsack-block-reexp-mic.x: knapsack-block-reexp-mic.o getoptions.o

#uts
#uts-base.x: uts-base.o brg_sha1.o uts.h head.h brg_types.h brg_sha1.h brg_endian.h

#uts-block-noreexp.x: uts-block-noreexp.o brg_sha1.o uts.h head.h brg_types.h brg_sha1.h brg_endian.h

#uts-block-reexp.x: uts-block-reexp.o brg_sha1.o uts.h head.h brg_types.h brg_sha1.h brg_endian.h 

#uts-block-noreexp-sse.x: uts-block-noreexp-sse.o brg_sha1-sse.o uts-sse.h head.h brg_types.h brg_sha1-sse.h brg_endian.h simd.h 

#uts-block-reexp-sse.x: uts-block-reexp-sse.o brg_sha1-sse.o uts-sse.h head.h brg_types.h brg_sha1-sse.h brg_endian.h simd.h 

#uts-block-noreexp-mic.x: uts-block-noreexp-mic.o brg_sha1-mic.o uts-mic.h head.h brg_types.h brg_sha1-mic.h brg_endian.h simd.h 

#uts-block-reexp-mic.x: uts-block-reexp-mic.o brg_sha1-mic.o uts-mic.h head.h brg_types.h brg_sha1-mic.h brg_endian.h simd.h 


############################################################################

.SUFFIXES: .cpp .x

.cpp.o:
	$(CC) $(CFLAGS) -I. -o $@ -c $<

%.x: %.o harness.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)
