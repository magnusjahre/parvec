The Recursive Benchmark Set for PLDI2015 Artifact Evaluation 
-------------------------------------------------------------

**************************************************************************************
*Version: 1.0
*Author:  Bin Ren, Youngjoon Jo, Sriram Krishnamoorthy, Kunal Agrawal, Milind Kulkarni
*Update:  Feb 13, 2015
*Paper:   Efficient Execution of Recursive Programs on Commodity Vector Hardware
*Benchs:  fib, binomial, nqueens, knapsack, graphcol, uts, parentheses, minmax
**************************************************************************************

The overall objective of this document is to help users quickly start
to use this benchmark set, and it contains the following descriptions:

0. Artifact evaluation expectations;
1. The evaluation environment;
2. The benchmark directory structure;
3. How to build the benchmarks;
4. How to run the benchmarks;
5. How to collect the numbers.

--------------------------------------------------------------------------------------

0. Artifact evaluation expectations
-----------------------------------

Our paper presents techniques for restructuring recursive,
task-parallel programs to run on vector hardware, and additional
transformations to maintain good vector utilization.  For artifact
evaluation, we present our benchmarks in their original form and in
several transformed forms, so that the reviewers can verify the
results presented in the paper, especially, Table 2, Figures 8 through
10, and Figures 13 through 16.  Other data can also be verified
(except Table 4, which is the performance report of our multi-core
versions).

In sum, by running the benchmarks built in the build-host, build-mic,
build-profile, build-distrib, build-reexp, and build-nostreamcompaction
directories, users can verify the following data in the paper:

  i) The execution time / speedup for each version
 ii) Cache miss profiling data
iii) SIMD utilization
 iv) Other profiling results


1. The evaluation environment
-----------------------------

This benchmark has been comprehensively tested in the following
environment:

a. For the CPU versions: Intel E5-2670 (8-core, 2.6GHz) Sandy Bridge,
32KB L1 Cache/Core, 20MB last level cache, 128-bit SSE4.2

b. For the Xeon Phi versions: Intel Xeon Phi 61-core SE10P
coprocessor (1.1 GHz), 32KB L1 Cache/Core, 512KB L2 Cache/Core,
512-bit AVX512

All codes are compiled with Intel icc-13.3.163 compiler with -O3
optimization (cpu flags: -O3 -DNDEBUG -unroll -DSSE41 -D__SSE4_1 -msse4.1).
The Xeon Phi codes are compiled with "-mmic" 
(mic flags: -O3 -funroll-loops -mmic), running in native mode. 
All codes are single core versions.

In case the users do not have Intel icc compiler available, the
sequential code without any SSE or AVX (target "plain" in the
rules.mk) can be built and run on CPU host only using the g++
compiler (make plain CC=g++).


2. The benchmark directory structure
------------------------------------

This benchmark set contains the following sub-directories:

a. benchmark application source codes (generically referred to as
   <benchname>): fib, binomial, nqueens, graphcol, knapsack, uts,
   parentheses, minmax.

   Each directory contains seven versions: 
   <benchname>-base             : base version;
   <benchname>-block-noreexp    : blocked breadth-first execution followed by depth-first
                                  without reexpansion optimization and without simd;
   <benchname>-block-reexp      : blocked breadth-first execution followed by depth-first
                                  with reexpansion optimization and without simd;
   <benchname>-block-noreexp-sse: blocked breadth-first execution followed by depth-first
                                  without reexpansion optimization and with sse;
   <benchname>-block-reexp-sse  : blocked breadth-first execution followed by depth-first
                                  with reexpansion optimization and with sse;
   <benchname>-block-noreexp-mic: blocked breadth-first execution followed by depth-first
                                  without reexpansion optimization and with avx512;
   <benchname>-block-reexp-mic  : blocked breadth-first execution followed by depth-first
                                  with reexpansion optimization and with avx512;

The sse versions are only executable on the CPU, while the mic
versions are only executable on the Xeon Phi. For all blocked
versions, we have corresponding "block-(sse|mic).h" header file to
provide the necessary "blocked software stack management code".  For
some benchmarks, such as knapsack and uts, we have some more files,
and the basic rule to organize these files is that the files ended
with "-sse"|"-mic" are the files that are only used by "sse"|"mic"
code, and the other files are shared by sequential codes.

b. Three shared utility directories: 
  common : some common header files for sequential and vectorized code;
  block  : some block profiler codes;
  harness: the common "main" function for all benchmarks.

c. An input directory -- inputs:
  data38N64E.col: used by graphcol, and users DON'T need to specify it during
                  the running of this code;
  knapsack-example1/2/3.input: used by knapsack, and users can use "-benchmark short",
                               "-benchmark medium" and "-benchmark long" to specify
                               the input, and this will be explained in Section 3.
  uts_test20.input: used by uts, and users need to specify it.

d. Six build directories for users to test our benchmarks (build the code at first):
  build-host: After build (see Section 3), all the CPU versions will be included
              in this directory. Users can test the absolute performance
              of each version to verify the efficacy of our "simd blocked"
              method. Please refer to Section 4 and 5 for more details.
  build-mic: Similar to build-host, this one contains all the XEON PHI versions.
  build-profile: This directory contains the versions with profile information.
                 By running this codes, users can mainly verify the SIMD
                 utilization data for CPU SEQUENTIAL BLOCK versions.
  build-distrib: This directory contains the CPU SEQUENTIAL BLOCK NOREEXP version for 
                 all benchmarks to collect the task distribution information.    
  build-reexp: This directory contains the CPU SEQUENTIAL BLOCK REEXP version for all
               benchmarks to collect the reexpansion benefit information. 
  build-nostreamcompaction: This directory contains the CPU and XEON PHI versions for
                            fib and nqueens to show the simd performance without
                            stream compaction optimization.

  NOTE: 1) If icc compiler and Xeon Phi is available, users can build all of
  these versions, otherwise, if users do not have icc compiler, please use 
  "make plain CC=g++", and users can run the CPU SEQUENTIAL BLOCK versions. 
  In such case, please only verify the "SIMD" utilization profiling data.  
  Such profiling is a sequential simulation, so we don't need any real simd support. 
  Please check Section 4,5 for details about running such profiling.
  2) The absolute performance reported in our paper is only for our specific
  execution environment, and if you have a different one, the performance may be
  different due to different compiler versions, CPU frequencies or cache sizes and
  cache behavior.  

e. The Makefile and related files for building: Makefile, common.mk, and rules.mk

3. How to build the benchmarks
-------------------------------

If icc compiler and a Xeon Phi coprocessor are available, please type
the make command in the benchmarks/ (root of the artifact package)
directory to build the executables:

$ make

It builds all versions in the build-host/, build-mic/, build-profile/, 
build-distrib/, build-reexp/, and build-nostreamcompaction/ directories.

Here are some recommendations if you don't have icc compiler or Xeon
Phi:

1) Without the icc compiler, users can NOT compile the SIMD versions
including sse and mic versions. For example, if you have only GNU g++
installed, please use the following command in benchmarks/ directory
to make the base version, and sequential blocked versions on CPU --
please NOTE: in such case, you can NOT verify the absolute performance
data in our paper, but can still verify the SIMD utilization (Figure
8):

$ make plain CC=g++ 


2) If users have icc compiler available, but don't have a Xeon Phi
coprocessor, please build as:

$ make cpu

These are all CPU versions. In this case, you can NOT verify the
performance data on MIC.


4. How to run the benchmarks
----------------------------

Below are instructions to execute these benchmarks from the command
line. We explain the procedure in two steps for each benchmark: 

step 1 -- the command line parameter format; 
step 2 -- the application parameters used in our paper as an example.

a. fib:

base version: ./fib-base.x <fib-number>
other versions: ./fib-XX.x <fib-number> <block-size n in terms of 2^n>
Example: ./fib-base.x 45 or ./fib-block-reexp-sse.x 45 9 
Here 45 is the fib-number used in our paper, while 9 denotes the block size is
2^9, i.e. 512, which is the best block size for this version -- Please check our
paper for more information about the best block sizes for other versions.

b. binomial:

base version: ./binomial-base <bin-large> <bin-small>
other versions: ./binomial-XX <bin-large> <bin-small> <block-size n in terms of 2^n>
Example: ./binomial-base.x 36 13 or ./binomial-block-reexp-sse.x 36 13 18 
Here 36, 13 are the application inputs used in our paper, while 18 denotes the
block size is 2^18 (262144) -- the best block size for this version.

c. nqueens:

base version: ./nqueens-base.x <#queens>
other versions: ./nqueens-XX.x <#queens> <block-size n in terms of 2^n>
Example: ./nqueens-base.x 13 or ./nqueens-block-reexp-sse.x 13 15 Here
13 is the number of queens used in our paper, while 15 denotes the
block size is 2^15 (32768) -- the best block size for this version.

d. graphcol:

base version: ./graphcol-base.x <#color>
other versions: ./graphcol-XX.x <#color> <block-size n in terms of 2^n>
Example: ./graphcol-base.x 3 or ./graphcol-block-reexp-sse.x 3 8 Here
3 means we are doing a 3-color problem used in our paper, while 8
denotes the block size is 2^8 (256) -- the best block size for this
version.

e. knapsack:

base version: ./knapsack-base.x -benchmark <input-test>
other versions: ./knapsack-XX.x -benchmark <input-test> -b <block-size n in 2^n>
Example: ./knapsack-base.x -benchmark long  
or ./knapsack-block-reexp-sse.x -benchmark long -b 11 
Here "-benchmark long" means that we use the largest data set
(inputs/knapsack-example3.input) from the cilk benchmark set in our
paper, while "-b 11" means that the block size is 2^11 (2048) -- the
best block size for this version. NOTE: we need to use "-b 11" rather
than "11".

f. uts:

base version: ./uts-base.x <input file> 
other versions: ./uts-XX.x <input file>  <block-size n>
Example: ./uts-base.x ../inputs/uts_test20.input
or ./uts-block-reexp-sse.x ../inputs/uts_test20.input 16384
Here ../inputs/uts_test20.input is our input configuration, and 16384
is the best block size for this version. NOTE: here the block size is
in the absolute value, rather than 2^n format. This is because we also
want to support some block size like 1000, 2000, and so on for this
benchmark.

g. parentheses:

base version: ./parentheses-base.x <parentheses-number>
other versions: ./parentheses-XX.x <parentheses-number> <block-size n in terms of 2^n>
Example: ./parentheses-base.x 19 or ./parentheses-block-reexp-sse.x 19
11 Here 19 is the number of generated parentheses pairs used in our
paper, while 11 denotes the block size is 2^11 (2048) -- the best
block size for this version.

h. minmax:

base version: ./minmax-base.x <#position>
other versions: ./minmax-XX.x <#position> <block-size n in terms of 2^n>
Example: ./minmax-base.x 12 or ./minmax-block-reexp-sse.x 12 10 Here
12 is the number of placed positions in this minmax game used in our
paper, while 10 denotes the block size is 2^10 (1024) -- the best
block size for this version.

We provide some simple scripts in build-host, build-mic and
build-profile directories: run-all-host.py, run-all-mic.py, and
run-all-profile.py.  (Similarly, we have some sample python scripts
for other build directories.)  Users can refer to these python scripts
to see how to run each benchmark. Please note, users may need some
modification to these scripts to achieve their own running objective.

5. How to collect the numbers
------------------------------

Below are detailed instructions for collecting the numbers presented
in the paper.

i) The execution time / speedup for each version

After building our benchmarks, please enter the build-host/ and/or
build-mic/ directories, and follow the above instructions to test the
absolute execution time of our codes, and to verify the efficacy of
our blocked simd execution.  Please notice: a) use the appropriate
block size (you can also choose different block sizes to see the
performance trend shown in our paper); and b) the sequential blocked
code cannot yield the performance benefit, and only the simd code can
show performance benefit, i.e., the performance curve in our paper
shows the blocked simd performance.

In sum, to test the execution time of our code, one should test:
<bench>-base, <bench>-block-noreexp-sse (similarly -mic) and
<bench>-block-reexp-sse (similarly -mic) versions and set up the correct
block size.  NOTE: In order to test the pure breadth first execution,
set the block size large enough for <bench>-block-noreexp-sse (similarly
-mic), i.e, fib:30, binomial:31, knapsack:31, nqueens:24, graphcol:25,
uts:14, parentheses:30, and minmax:26.


ii) The cache miss profiling data 

To get the cache miss profiling data as shown in our paper, please
also enter the build-host/ and/or build-mic/ directories, and test on
<bench>-base, <bench>-block-noreexp-(sse|mic) and
<bench>-block-reexp-(sse|mic), three versions.

Our work uses two kinds of profilers: valgrind/cachegrind for CPU, and
Intel Vtune for Xeon Phi, and here are two examples for fib:

on CPU (build-host):
valgrind --tool=cachegrind  ./fib-block-reexp-sse.x 45 10
(You can also specify the result place: --cachegrind-out-file=resultX)

on MIC (build-mic):
amplxe-cl -collect knc-general-exploration ./fib-block-reexp-mic.x 45 10
(You can also specify the result place: -result-dir resultX )


iii) The simd utilization.

The simd utilization is data from a sequential simulation. You can get
this data without any real simd execution. In other words, if users
have only g++ installed, this data is the most (or even the only)
reasonable one to verify.

Please enter the directory: build-profile,
and test on two versions: <bench>-block-noreexp.x and <bench>-block-reexp.x.

For all benchmarks except knapsack, we profile the simd utilization
with simd width 16, and for knapsack, we profile the simd width 8. By
varying the block size, we can get some curves as shown in our paper.

NOTE: According to our paper, we define the simd utilization as the
work percentage executed with full simd width, i.e., the number
followed "work 16:" (or "work 8:" for knapsack) in the profiling
output.

iv) Other profiling results.

Our paper also shows some other profiling data, such as the
distribution of tasks for each benchmark, the benefit of re-expansion,
and the speedup with and without stream compaction.
Here are some simple instructions to get such profiling information:

a. distribution of tasks: enter the build-distrib/ directory and run
   <bench>-block-noreexp.x code to get such information. When running
   this code, it is better to set the block size as the largest one,
   i.e, fib:30, binomial:31, knapsack:31, nqueens:24, graphcol:25,
   uts:14, parentheses:30, and minmax:26.

b. benefit of re-expansion: enter the build-reexp/ directory and run
   <bench>-block-reexp.x code to get such information. When running
   this code, please use the best block size for CPU, i.e., fib:9,
   binomial:18, nqueens:15, graphcol:8, knapsack:11, uts:14,
   parentheses:11, minmax:10.

c. stream compaction benefit: test only for fib and nqueens, for sse
   and mic versions. Enter the build-nostreamcompaction/ directory and
   run (fib|nqueens)-block-(noreexp|reexp)-(sse|mic).x code to get the
   simd performance without stream compaction.

Please contact us by the e-mail address below in case you have any further
questions on running this code.

**********************************
Contact: Bin Ren bin.ren@pnnl.gov
**********************************

