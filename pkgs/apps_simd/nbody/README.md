# nbody
n-body solver, with the aim of writing a fast MPI implementation.

For the n-body problem this is a little non-trivial since each timestep the position of every body is needed and we need to do a reduction of the acceleration contributions, so it is something of a communication-heavy problem. Cluster available for testing has only a Gigabit Ethernet interconnect, so efficient communication will be essential.

For each pair of bodies, we need to compute Fij = G\*mi\*mj\*(ri-rj)/|ri-rj|^3. Since the forces are equal and opposite, we only have to compute the upper triangle of this matrix. The forces give us acceleration values, which we use to update the velocity and position using a simple Euler integration.

For n bodies, then, we need to perform n\*(n/2) force calculations. We expect performance measured in "Timesteps/Second" to scale in this way.


### Serial Implementation
Body position, velocity, acceleration and mass are each stored in separate arrays rather than a somewhat neater struct-based approach, to make it easier for the auto-vectorizer and to use manual vector intrinsics.

#### Manual Vectorization
Compute accelerations for 2, 4 body pairs at a time (double precision, SSE, AVX) in inner loop of `ComputeAccel`, routine called `ComputeAccelVec`. This is twice as fast as the autovectorized version using SSE, but only slightly faster than this with AVX!

The reason for this is possibly that the inner loop of `ComputeAccelVec` includes the instructions (V)SQRTPD and (V)DIVPD. These are very high latency instructions which account for 67% of the loop's total cycles. On Intel chips, the AVX instruction has twice the latency of the corresponding SSE instruction, so is no faster. On AMD's Piledriver, the latencies for VSQRTPD and VDIVPD are only 1.93 and 1.8 times longer, respectively. These instructions should execute ~15% faster Piledriver with AVX, then, and actually this is more or less the performance improvement that we see, even though this is only 67% of the loop.

Additionally, these instructions on Intel Ivy-Bridge CPUs have only 64% of the latency compared to Sandy-Bridge. We see that the 4GHz E5-2667v2 (Ivy-Bridge) is significantly faster than the 4.4GHz i5-2500K (Sandy-Bridge).

![Serial Performance](plots/img/0-plot.png)

The Opteron 6128 system is what we will use for MPI testing across multiple nodes.


### OpenMP
Before starting with MPI, we can check how OpenMP code scales with core-count. This gives us something to aim for when running an MPI implementation on a shared memory system. Accounting for the difference in turbo-boost clocks between a single and 4-thread load (i5-2500K) the optimal speedup is 3.82x.

For best performance the OpenMP scheduling needs to be set properly. Since we are only computing the upper triangle of the force matrix, later loops are much faster since they have less work to do. For this reason, we choose `static` scheduling with as small a "chunk-size" as possible (but no smaller than 8, we don't want to induce cache misses and false sharing by making chunks smaller than one cache line).

![OpenMP Scaling](plots/img/3-scaling-sse.png)

For large numbers of bodies scaling is very good, with most configurations reaching 90% optimal scaling after around 250 particles/thread. Note there is no 16C result plotted for the Opteron 6328 -- these CPUs have only a single FPU per pair of cores, so the system does not scale at all above 8C.

Also, at the maximum core counts, we benefit from making NUMA-friendly memory allocations, achieved here simply by parallelising the initialization routine to expolit the "first touch" policy used by the Linux OS.

On a shared memory system, then, we expect a good MPI implementation also to achieve near-optimal scaling with core count for large numbers of bodies.


### MPI
We initially choose simply to split the particles evenly among the MPI ranks. When using OpenMP we saw that it was important to choose the scheduling properly to obtain good scaling, so this will not be well-performing code.

The procedure is as follows:
*	Master thread generates initial conditions, and then sends the position, acceleration and mass of all bodies to all threads, and the velocity of the bodies to the appropriate threads.
*	Each thread computes acceleration contributions for pairings between its share of bodies and all other bodies.
*	Global reduction of acceleration contributions using `MPI_Allreduce`.
*	Each thread updates its bodies positions, and uses `MPI_Bcast` to send the new positions to all other threads.

As expected, the performance is poor for this work-sharing scheme (marked `Equal` in the plot).

![Equal sharing, Vampir](plots/img/4-badloadbalance.png)

Using VampirTrace and Vampir, we can see that threads computing accelerations from particles near the end of the loop finish very quickly, and block in `MPI_Allreduce`, waiting for the other threads to complete their work. The actual reduction takes very little time.

#### Better load-balancing
We should share the work evenly among the threads. We have to run the `ComputeAccel` loop body n^2/2 times, so if we have p threads, we want each to do n^2/(2p) of these. This can be achieved by having each thread run up to body number n-(p+Sqrt((-1+n) p (-(rank+1)n+(n-1)p)))/p, suitably rounded, from the maximum of the rank below it (from body 0, for rank 0).

![Balanced sharing, Vampir](plots/img/4-goodloadbalance.png)

This scheme seems to have worked well. With 16 threads (no SSE!) we now spend around 5.5% of the time communicating, and the scaling with thread count is much better.

![MPI Scaling](plots/img/4-plot.png)

#### More than one node...
The cluster available for testing has only gigabit ethernet interconnects. As soon as we start splitting ranks across nodes, performance degrades. If we run 16 threads (no SSE!) as above but with 1 thread on each of 16 nodes, communication accounts for some 45% of the runtime:

![16 nodes, Vampir](plots/img/5-poorcomms.png)

We can try to run with a larger problem size. The computation scales with nBodies^2, whereas the data transfer scales linearly. This allows more than one node to be of benefit for larger problem sizes, but the scaling is still not very good:

![MPI Scaling](plots/img/5-plot.png)

The problem is that we are saturating the gigabit links between nodes after ~32 threads.

The first thing to realize is that the `UpdatePositions` function takes almost no time at all. It would be far better (since we are using `MPI_Allreduce`, so all ranks know the full acceleration array) if each rank were to update its own position arrays and avoid the broadcast entirely. This dramatically cuts the network usage, and we start to see some reasonable scaling:

![MPI Scaling](plots/img/5-plot-nor.png)

The remaining bottleneck is `MPI_Allreduce`. This cannot usefully be overlapped with any computation, so the only way to make any further gains is to reduce the number of ranks taking part in the reduction.


### Hybrid MPI + OpenMP
This can be achieved with a mixture of MPI and OpenMP. The goal is to run fewer MPI threads per node, with OpenMP threading loops within each rank's section of particles. Whether it is best to run a single MPI thread per node, or to run more needs to be investigated.

![Hybrid Runtimes](plots/img/6-hybrid-times.png)

Wow! That didn't help at all. What we should bear in mind is that MPI implementations aren't stupid (mostly?) They are aware that some communication is intra-node, and use shared memory accordingly here. By adding OpenMP we have added extra overhead, and not reduced the network load at all.

That said, what we have learnt from this is that if you run a hybrid code, it can be important to account of NUMA effects, and bind threads accordingly.











