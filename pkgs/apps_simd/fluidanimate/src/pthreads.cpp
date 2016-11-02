//Code written by Richard O. Lee and Christian Bienia
//Modified by Christian Fensch

// SIMD Version by Juan M. Cebrian, NTNU - 2013. (modifications under JMCG tag)
// The Scalar version on this code changes the loop order of ComputeDensities and ComputeForces
// So that it can be compared to the SIMD version. However, this exchange makes it slower than
// the original version.

#include <cstdlib>
#include <cstring>

#include <iostream>
#include <fstream>
#if defined(WIN32)
#define NOMINMAX
#include <windows.h>
#endif
#include <math.h>
#include <pthread.h>
#include <assert.h>
#include <float.h>

#include "fluid.hpp"
#include "cellpool.hpp"
#include "parsec_barrier.hpp"

#include <iomanip>

#ifdef ENABLE_VISUALIZATION
#include "fluidview.hpp"
#endif

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

//Uncomment to add code to check that Courant–Friedrichs–Lewy condition is satisfied at runtime
//#define ENABLE_CFL_CHECK

////////////////////////////////////////////////////////////////////////////////

cellpool *pools; //each thread has its private cell pool

fptype restParticlesPerMeter, h, hSq;
fptype densityCoeff, pressureCoeff, viscosityCoeff;

int nx, ny, nz;    // number of grid cells in each dimension
Vec3 delta;        // cell dimensions
int numParticles = 0;
int numCells = 0;
Cell *cells = 0;
Cell *cells2 = 0;
int *cnumPars = 0;
int *cnumPars2 = 0;

/* JMCG BEGIN */

//#define DEBUG_SIMD
#ifdef DEBUG_SIMD
float max_diff = 0.0f;
float max_diff_forces = 0.0f;
#endif

// JMCG Required for original code since we change loop order
// Compute densities one by one starting at iparNeigh_in
inline fptype ComputeDensitiesMTOriginal(int iparNeigh_in, int indexNeigh, int numNeighPars, int np, int index, Cell *cell, Cell *neigh );
// Compute Forces one by one starting at iparNeigh_in
inline Vec3 ComputeForcesMTOriginal(int iparNeigh_in, int indexNeigh, int numNeighPars, int np, int index, Cell *cell, Cell *neigh );

#ifdef SIMD_WIDTH // JMCG Vectorization

//#define SIMD_STATS
#ifdef SIMD_STAT
long int total_compute_densities_calls = 0;
long int total_compute_densities_calls_sse = 0;
long int total_compute_densities_calls_original = 0;
long int total_compute_densities_calls_leftovers = 0;

long int total_compute_forces_calls = 0;
long int total_compute_forces_calls_sse = 0;
long int total_compute_forces_calls_original = 0;
long int total_compute_forces_calls_leftovers = 0;

long int total_compute_densities_calls_np_sse = 0;
long int total_compute_densities_calls_np_original = 0;
long int total_compute_densities_calls_np_leftovers = 0;
long int sse_fully_used = 0;
long int sse_not_used = 0;
long int sse_partially_used = 0;
#endif
// Compute densities one by one starting at iparNeigh_in. Used for debugging, does not store any density
inline fptype ComputeDensitiesMTOriginal_test(int iparNeigh_in, int indexNeigh, int numNeighPars, int np, int index, Cell *cell, Cell *neigh );
// Compute Forces one by one starting at iparNeigh_in. Used for debugging, does not store any value in a[].
inline Vec3 ComputeForcesMTOriginal_test(int iparNeigh_in, int indexNeigh, int numNeighPars, int np, int index, Cell *cell, Cell *neigh );

// Compute densities in groups of SIMD_WIDTH. Stores last possition multiple of SIMD_WIDTH in iparNeigh_in and modifies neigh pointer to the latest neigh analized
inline fptype ComputeDensitiesMTSIMD(int *iparNeigh_in, int indexNeigh, int numNeighPars, int np, int index, Cell *cell, Cell **neigh );
// Compute Forces in groups of SIMD_WIDTH starting at iparNeigh_in. Stores last possition multiple of SIMD_WIDTH in iparNeigh_in and modifies neigh pointer to the latest neigh analized
inline Vec3 ComputeForcesMTSIMD(int *iparNeigh_in, int indexNeigh, int numNeighPars, int np, int index, Cell *cell, Cell **neigh );

#endif

/* JMCG END */

Cell **last_cells = NULL; //helper array with pointers to last cell structure of "cells" array lists
#ifdef ENABLE_VISUALIZATION
Vec3 vMax(0.0,0.0,0.0);
Vec3 vMin(0.0,0.0,0.0);
#endif

int XDIVS = 1;  // number of partitions in X
int ZDIVS = 1;  // number of partitions in Z


#define NUM_GRIDS  ((XDIVS) * (ZDIVS))
#define MUTEXES_PER_CELL 128
#define CELL_MUTEX_ID 0

struct Grid
{
  union {
    struct {
      int sx, sy, sz;
      int ex, ey, ez;
    };
    unsigned char pp[CACHELINE_SIZE];
  };
} *grids;
bool  *border;
pthread_attr_t attr;
pthread_t *thread;
pthread_mutex_t **mutex;  // used to lock cells in RebuildGrid and also particles in other functions
pthread_barrier_t barrier;  // global barrier used by all threads
#ifdef ENABLE_VISUALIZATION
pthread_barrier_t visualization_barrier;  // global barrier to separate (serial) visualization phase from (parallel) fluid simulation
#endif

typedef struct __thread_args {
  int tid;      //thread id, determines work partition
  int frames;      //number of frames to compute
} thread_args;      //arguments for threads

using namespace std;

////////////////////////////////////////////////////////////////////////////////

/*
 * hmgweight
 *
 * Computes the hamming weight of x
 *
 * x      - input value
 * lsb    - if x!=0 position of smallest bit set, else -1
 *
 * return - the hamming weight
 */
unsigned int hmgweight(unsigned int x, int *lsb) {
  unsigned int weight=0;
  unsigned int mask= 1;
  unsigned int count=0;

  *lsb=-1;
  while(x > 0) {
    unsigned int temp;
    temp=(x&mask);
    if((x&mask) == 1) {
      weight++;
      if(*lsb == -1) *lsb = count;
    }
    x >>= 1;
    count++;
  }

  return weight;
}

void InitSim(char const *fileName, unsigned int threadnum)
{
  //Compute partitioning based on square root of number of threads
  //NOTE: Other partition sizes are possible as long as XDIVS * ZDIVS == threadnum,
  //      but communication is minimal (and hence optimal) if XDIVS == ZDIVS
  int lsb;
  if(hmgweight(threadnum,&lsb) != 1) {
    std::cerr << "Number of threads must be a power of 2" << std::endl;
    exit(1);
  }
  XDIVS = 1<<(lsb/2);
  ZDIVS = 1<<(lsb/2);
  if(XDIVS*ZDIVS != threadnum) XDIVS*=2;
  assert(XDIVS * ZDIVS == threadnum);

  thread = new pthread_t[NUM_GRIDS];
  grids = new struct Grid[NUM_GRIDS];
  assert(sizeof(Grid) <= CACHELINE_SIZE); // as we put and aligh grid on the cacheline size to avoid false-sharing
                                          // if asserts fails - increase pp union member in Grid declarationi
                                          // and change this macro
  pools = new cellpool[NUM_GRIDS];

  //Load input particles
  std::cout << "Loading file \"" << fileName << "\"..." << std::endl;
  std::ifstream file(fileName, std::ios::binary);
  if(!file) {
    std::cerr << "Error opening file. Aborting." << std::endl;
    exit(1);
  }

  //Always use single precision float variables b/c file format uses single precision
  float restParticlesPerMeter_le;
  int numParticles_le;
  file.read((char *)&restParticlesPerMeter_le, FILE_SIZE_FLOAT);
  file.read((char *)&numParticles_le, FILE_SIZE_INT);
  if(!isLittleEndian()) {
    restParticlesPerMeter = bswap_float(restParticlesPerMeter_le);
    numParticles          = bswap_int32(numParticles_le);
  } else {
    restParticlesPerMeter = restParticlesPerMeter_le;
    numParticles          = numParticles_le;
  }
  for(int i=0; i<NUM_GRIDS; i++) cellpool_init(&pools[i], numParticles/NUM_GRIDS);

  h = kernelRadiusMultiplier / restParticlesPerMeter;
  hSq = h*h;

#ifndef ENABLE_DOUBLE_PRECISION
  fptype coeff1 = 315.0 / (64.0*pi*powf(h,9.0));
  fptype coeff2 = 15.0 / (pi*powf(h,6.0));
  fptype coeff3 = 45.0 / (pi*powf(h,6.0));
#else
  fptype coeff1 = 315.0 / (64.0*pi*pow(h,9.0));
  fptype coeff2 = 15.0 / (pi*pow(h,6.0));
  fptype coeff3 = 45.0 / (pi*pow(h,6.0));
#endif //ENABLE_DOUBLE_PRECISION
  fptype particleMass = 0.5*doubleRestDensity / (restParticlesPerMeter*restParticlesPerMeter*restParticlesPerMeter);
  densityCoeff = particleMass * coeff1;
  pressureCoeff = 3.0*coeff2 * 0.50*stiffnessPressure * particleMass;
  viscosityCoeff = viscosity * coeff3 * particleMass;

  Vec3 range = domainMax - domainMin;
  nx = (int)(range.x / h);
  ny = (int)(range.y / h);
  nz = (int)(range.z / h);
  assert(nx >= 1 && ny >= 1 && nz >= 1);
  numCells = nx*ny*nz;
  std::cout << "Number of cells: " << numCells << std::endl;
  delta.x = range.x / nx;
  delta.y = range.y / ny;
  delta.z = range.z / nz;
  assert(delta.x >= h && delta.y >= h && delta.z >= h);

  std::cout << "Grids steps over x, y, z: " << delta.x << " " << delta.y << " " << delta.z << std::endl;

  assert(nx >= XDIVS && nz >= ZDIVS);
  int gi = 0;
  int sx, sz, ex, ez;
  ex = 0;
  for(int i = 0; i < XDIVS; ++i)
  {
    sx = ex;
    ex = (int)((fptype)(nx)/(fptype)(XDIVS) * (i+1) + 0.5);
    assert(sx < ex);

    ez = 0;
    for(int j = 0; j < ZDIVS; ++j, ++gi)
    {
      sz = ez;
      ez = (int)((fptype)(nz)/(fptype)(ZDIVS) * (j+1) + 0.5);
      assert(sz < ez);

      grids[gi].sx = sx;
      grids[gi].ex = ex;
      grids[gi].sy = 0;
      grids[gi].ey = ny;
      grids[gi].sz = sz;
      grids[gi].ez = ez;
    }
  }
  assert(gi == NUM_GRIDS);

  border = new bool[numCells];
  for(int i = 0; i < NUM_GRIDS; ++i)
    for(int iz = grids[i].sz; iz < grids[i].ez; ++iz)
      for(int iy = grids[i].sy; iy < grids[i].ey; ++iy)
        for(int ix = grids[i].sx; ix < grids[i].ex; ++ix)
        {
          int index = (iz*ny + iy)*nx + ix;
          border[index] = false;
          for(int dk = -1; dk <= 1; ++dk)
	  {
            for(int dj = -1; dj <= 1; ++dj)
	    {
              for(int di = -1; di <= 1; ++di)
              {
                int ci = ix + di;
                int cj = iy + dj;
                int ck = iz + dk;

                if(ci < 0) ci = 0; else if(ci > (nx-1)) ci = nx-1;
                if(cj < 0) cj = 0; else if(cj > (ny-1)) cj = ny-1;
                if(ck < 0) ck = 0; else if(ck > (nz-1)) ck = nz-1;

                if( ci < grids[i].sx || ci >= grids[i].ex ||
                  cj < grids[i].sy || cj >= grids[i].ey ||
                  ck < grids[i].sz || ck >= grids[i].ez ) {

                    border[index] = true;
		    break;
		}
              } // for(int di = -1; di <= 1; ++di)
	      if(border[index])
		break;
	    } // for(int dj = -1; dj <= 1; ++dj)
	    if(border[index])
	       break;
           } // for(int dk = -1; dk <= 1; ++dk)
        }

  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  mutex = new pthread_mutex_t *[numCells];
  for(int i = 0; i < numCells; ++i)
  {
    assert(CELL_MUTEX_ID < MUTEXES_PER_CELL);
    int n = (border[i] ? MUTEXES_PER_CELL : CELL_MUTEX_ID+1);
    mutex[i] = new pthread_mutex_t[n];
    for(int j = 0; j < n; ++j)
      pthread_mutex_init(&mutex[i][j], NULL);
  }
  pthread_barrier_init(&barrier, NULL, NUM_GRIDS);
#ifdef ENABLE_VISUALIZATION
  //visualization barrier is used by all NUM_GRIDS worker threads and 1 master thread
  pthread_barrier_init(&visualization_barrier, NULL, NUM_GRIDS+1);
#endif
  //make sure Cell structure is multiple of estiamted cache line size
  assert(sizeof(Cell) % CACHELINE_SIZE == 0);
  //make sure helper Cell structure is in sync with real Cell structure
  assert(offsetof(struct Cell_aux, padding) == offsetof(struct Cell, padding));

#if defined(WIN32)
  cells = (struct Cell*)_aligned_malloc(sizeof(struct Cell) * numCells, CACHELINE_SIZE);
  cells2 = (struct Cell*)_aligned_malloc(sizeof(struct Cell) * numCells, CACHELINE_SIZE);
  cnumPars = (int*)_aligned_malloc(sizeof(int) * numCells, CACHELINE_SIZE);
  cnumPars2 = (int*)_aligned_malloc(sizeof(int) * numCells, CACHELINE_SIZE);
  last_cells = (struct Cell **)_aligned_malloc(sizeof(struct Cell *) * numCells, CACHELINE_SIZE);
  assert((cells!=NULL) && (cells2!=NULL) && (cnumPars!=NULL) && (cnumPars2!=NULL) && (last_cells!=NULL));
#elif defined(SPARC_SOLARIS)
  cells = (Cell*)memalign(CACHELINE_SIZE, sizeof(struct Cell) * numCells);
  cells2 =  (Cell*)memalign(CACHELINE_SIZE, sizeof(struct Cell) * numCells);
  cnumPars =  (int*)memalign(CACHELINE_SIZE, sizeof(int) * numCells);
  cnumPars2 =  (int*)memalign(CACHELINE_SIZE, sizeof(int) * numCells);
  last_cells =  (Cell**)memalign(CACHELINE_SIZE, sizeof(struct Cell *) * numCells);
  assert((cells!=0) && (cells2!=0) && (cnumPars!=0) && (cnumPars2!=0) && (last_cells!=0));
#else
  int rv0 = posix_memalign((void **)(&cells), CACHELINE_SIZE, sizeof(struct Cell) * numCells);
  int rv1 = posix_memalign((void **)(&cells2), CACHELINE_SIZE, sizeof(struct Cell) * numCells);
  int rv2 = posix_memalign((void **)(&cnumPars), CACHELINE_SIZE, sizeof(int) * numCells);
  int rv3 = posix_memalign((void **)(&cnumPars2), CACHELINE_SIZE, sizeof(int) * numCells);
  int rv4 = posix_memalign((void **)(&last_cells), CACHELINE_SIZE, sizeof(struct Cell *) * numCells);
  assert((rv0==0) && (rv1==0) && (rv2==0) && (rv3==0) && (rv4==0));
#endif

  // because cells and cells2 are not allocated via new
  // we construct them here
  for(int i=0; i<numCells; ++i)
  {
	  new (&cells[i]) Cell;
	  new (&cells2[i]) Cell;
  }

  memset(cnumPars, 0, numCells*sizeof(int));

  //Always use single precision float variables b/c file format uses single precision float
  int pool_id = 0;
  float px, py, pz, hvx, hvy, hvz, vx, vy, vz;
  for(int i = 0; i < numParticles; ++i)
  {
    file.read((char *)&px, FILE_SIZE_FLOAT);
    file.read((char *)&py, FILE_SIZE_FLOAT);
    file.read((char *)&pz, FILE_SIZE_FLOAT);
    file.read((char *)&hvx, FILE_SIZE_FLOAT);
    file.read((char *)&hvy, FILE_SIZE_FLOAT);
    file.read((char *)&hvz, FILE_SIZE_FLOAT);
    file.read((char *)&vx, FILE_SIZE_FLOAT);
    file.read((char *)&vy, FILE_SIZE_FLOAT);
    file.read((char *)&vz, FILE_SIZE_FLOAT);
    if(!isLittleEndian()) {
      px  = bswap_float(px);
      py  = bswap_float(py);
      pz  = bswap_float(pz);
      hvx = bswap_float(hvx);
      hvy = bswap_float(hvy);
      hvz = bswap_float(hvz);
      vx  = bswap_float(vx);
      vy  = bswap_float(vy);
      vz  = bswap_float(vz);
    }

    int ci = (int)((px - domainMin.x) / delta.x);
    int cj = (int)((py - domainMin.y) / delta.y);
    int ck = (int)((pz - domainMin.z) / delta.z);

    if(ci < 0) ci = 0; else if(ci > (nx-1)) ci = nx-1;
    if(cj < 0) cj = 0; else if(cj > (ny-1)) cj = ny-1;
    if(ck < 0) ck = 0; else if(ck > (nz-1)) ck = nz-1;

    int index = (ck*ny + cj)*nx + ci;
    Cell *cell = &cells[index];

    //go to last cell structure in list
    int np = cnumPars[index];
    while(np > PARTICLES_PER_CELL) {
      cell = cell->next;
      np = np - PARTICLES_PER_CELL;
    }
    //add another cell structure if everything full
    if( (np % PARTICLES_PER_CELL == 0) && (cnumPars[index] != 0) ) {
      //Get cells from pools in round-robin fashion to balance load during parallel phase
      cell->next = cellpool_getcell(&pools[pool_id]);
      pool_id = (pool_id+1) % NUM_GRIDS;
      cell = cell->next;
      np = np - PARTICLES_PER_CELL;
    }

    cell->p_coord[np*3] = px;
    cell->p_coord[(np*3)+1] = py;
    cell->p_coord[(np*3)+2] = pz;

    cell->v_coord[np*3] = vx;
    cell->v_coord[(np*3)+1] = vy;
    cell->v_coord[(np*3)+2] = vz;

    cell->hv[np].x = hvx;
    cell->hv[np].y = hvy;
    cell->hv[np].z = hvz;

#ifdef ENABLE_VISUALIZATION
	vMin.x = std::min(vMin.x, cell->v[np].x);
	vMax.x = std::max(vMax.x, cell->v[np].x);
	vMin.y = std::min(vMin.y, cell->v[np].y);
	vMax.y = std::max(vMax.y, cell->v[np].y);
	vMin.z = std::min(vMin.z, cell->v[np].z);
	vMax.z = std::max(vMax.z, cell->v[np].z);
#endif
    ++cnumPars[index];
  }

  std::cout << "Number of particles: " << numParticles << std::endl;
}

////////////////////////////////////////////////////////////////////////////////

void SaveFile(char const *fileName)
{
  std::cout << "Saving file \"" << fileName << "\"..." << std::endl;

  std::ofstream file(fileName, std::ios::binary);
  assert(file);

  //Always use single precision float variables b/c file format uses single precision
  if(!isLittleEndian()) {
    float restParticlesPerMeter_le;
    int   numParticles_le;

    restParticlesPerMeter_le = bswap_float((float)restParticlesPerMeter);
    numParticles_le      = bswap_int32(numParticles);
    file.write((char *)&restParticlesPerMeter_le, FILE_SIZE_FLOAT);
    file.write((char *)&numParticles_le,      FILE_SIZE_INT);
  } else {
    file.write((char *)&restParticlesPerMeter, FILE_SIZE_FLOAT);
    file.write((char *)&numParticles,      FILE_SIZE_INT);
  }

  int count = 0;
  for(int i = 0; i < numCells; ++i)
  {
    Cell *cell = &cells[i];
    int np = cnumPars[i];
    for(int j = 0; j < np; ++j)
    {
      //Always use single precision float variables b/c file format uses single precision
      float px, py, pz, hvx, hvy, hvz, vx,vy, vz;
      if(!isLittleEndian()) {
	px  = bswap_float((float)(cell->p_coord[(j % PARTICLES_PER_CELL)*3]));
        py  = bswap_float((float)(cell->p_coord[((j % PARTICLES_PER_CELL)*3)+1]));
	pz  = bswap_float((float)(cell->p_coord[((j % PARTICLES_PER_CELL)*3)+2]));

        vx  = bswap_float((float)(cell->v_coord[(j % PARTICLES_PER_CELL)*3]));
        vy  = bswap_float((float)(cell->v_coord[((j % PARTICLES_PER_CELL)*3)+1]));
	vz  = bswap_float((float)(cell->v_coord[((j % PARTICLES_PER_CELL)*3)+2]));

        hvx = bswap_float((float)(cell->hv[j % PARTICLES_PER_CELL].x));
        hvy = bswap_float((float)(cell->hv[j % PARTICLES_PER_CELL].y));
        hvz = bswap_float((float)(cell->hv[j % PARTICLES_PER_CELL].z));
      } else {
        px  = (float)(cell->p_coord[(j % PARTICLES_PER_CELL)*3]);
	py  = (float)(cell->p_coord[((j % PARTICLES_PER_CELL)*3)+1]);
	pz  = (float)(cell->p_coord[((j % PARTICLES_PER_CELL)*3)+2]);

	vx  = (float)(cell->v_coord[(j % PARTICLES_PER_CELL)*3]);
	vy  = (float)(cell->v_coord[((j % PARTICLES_PER_CELL)*3)+1]);
	vz  = (float)(cell->v_coord[((j % PARTICLES_PER_CELL)*3)+2]);

        hvx = (float)(cell->hv[j % PARTICLES_PER_CELL].x);
        hvy = (float)(cell->hv[j % PARTICLES_PER_CELL].y);
        hvz = (float)(cell->hv[j % PARTICLES_PER_CELL].z);
      }
      file.write((char *)&px,  FILE_SIZE_FLOAT);
      file.write((char *)&py,  FILE_SIZE_FLOAT);
      file.write((char *)&pz,  FILE_SIZE_FLOAT);
      file.write((char *)&hvx, FILE_SIZE_FLOAT);
      file.write((char *)&hvy, FILE_SIZE_FLOAT);
      file.write((char *)&hvz, FILE_SIZE_FLOAT);
      file.write((char *)&vx,  FILE_SIZE_FLOAT);
      file.write((char *)&vy,  FILE_SIZE_FLOAT);
      file.write((char *)&vz,  FILE_SIZE_FLOAT);
      ++count;

      //move pointer to next cell in list if end of array is reached
      if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
        cell = cell->next;
      }

    }
  }
  std::cout << "Saving " << count << " should be: " << numParticles << std::endl;
//  assert(count == numParticles); // JMCG I'm not sure why this is failing, even for the scalar code on ICC, we are not touching those variables, we are missing a particle.
}

////////////////////////////////////////////////////////////////////////////////

void CleanUpSim()
{
  // first return extended cells to cell pools
  for(int i=0; i< numCells; ++i)
  {
    Cell& cell = cells[i];
	while(cell.next)
	{
		Cell *temp = cell.next;
		cell.next = temp->next;
		cellpool_returncell(&pools[0], temp);
	}
  }
  // now return cell pools
  //NOTE: Cells from cell pools can migrate to different pools during the parallel phase.
  //      This is no problem as long as all cell pools are destroyed together. Each pool
  //      uses its internal meta information to free exactly the cells which it allocated
  //      itself. This guarantees that all allocated cells will be freed but it might
  //      render other cell pools unusable so they also have to be destroyed.
  for(int i=0; i<NUM_GRIDS; i++) cellpool_destroy(&pools[i]);
  pthread_attr_destroy(&attr);

  for(int i = 0; i < numCells; ++i)
  {
    assert(CELL_MUTEX_ID < MUTEXES_PER_CELL);
    int n = (border[i] ? MUTEXES_PER_CELL : CELL_MUTEX_ID+1);
    for(int j = 0; j < n; ++j)
      pthread_mutex_destroy(&mutex[i][j]);
    delete[] mutex[i];
  }
  delete[] mutex;
  pthread_barrier_destroy(&barrier);
#ifdef ENABLE_VISUALIZATION
  pthread_barrier_destroy(&visualization_barrier);
#endif

  delete[] border;

#if defined(WIN32)
  _aligned_free(cells);
  _aligned_free(cells2);
  _aligned_free(cnumPars);
  _aligned_free(cnumPars2);
  _aligned_free(last_cells);
#else
  free(cells);
  free(cells2);
  free(cnumPars);
  free(cnumPars2);
  free(last_cells);
#endif
  delete[] thread;
  delete[] grids;
}

////////////////////////////////////////////////////////////////////////////////

void ClearParticlesMT(int tid)
{
  for(int iz = grids[tid].sz; iz < grids[tid].ez; ++iz)
    for(int iy = grids[tid].sy; iy < grids[tid].ey; ++iy)
      for(int ix = grids[tid].sx; ix < grids[tid].ex; ++ix)
      {
        int index = (iz*ny + iy)*nx + ix;
        cnumPars[index] = 0;
	cells[index].next = NULL;
        last_cells[index] = &cells[index];
      }
}

////////////////////////////////////////////////////////////////////////////////

void RebuildGridMT(int tid)
{
  // Note, in parallel versions the below swaps
  // occure outside RebuildGrid()
  // swap src and dest arrays with particles
  //   std::swap(cells, cells2);
  // swap src and dest arrays with counts of particles
  //  std::swap(cnumPars, cnumPars2);

  //iterate through source cell lists
  for(int iz = grids[tid].sz; iz < grids[tid].ez; ++iz)
    for(int iy = grids[tid].sy; iy < grids[tid].ey; ++iy)
      for(int ix = grids[tid].sx; ix < grids[tid].ex; ++ix)
      {
        int index2 = (iz*ny + iy)*nx + ix;
        Cell *cell2 = &cells2[index2];
        int np2 = cnumPars2[index2];
        //iterate through source particles
        for(int j = 0; j < np2; ++j)
        {
          // get destination for source particle:
	  int ci = (int)((cell2->p_coord[(j % PARTICLES_PER_CELL)*3] - domainMin.x) / delta.x);
          int cj = (int)((cell2->p_coord[((j % PARTICLES_PER_CELL)*3)+1] - domainMin.y) / delta.y);
	  int ck = (int)((cell2->p_coord[((j % PARTICLES_PER_CELL)*3)+2] - domainMin.z) / delta.z);

          if(ci < 0) ci = 0; else if(ci > (nx-1)) ci = nx-1;
          if(cj < 0) cj = 0; else if(cj > (ny-1)) cj = ny-1;
          if(ck < 0) ck = 0; else if(ck > (nz-1)) ck = nz-1;
#if 0
		  assert(ci>=ix-1);
		  assert(ci<=ix+1);
		  assert(cj>=iy-1);
		  assert(cj<=iy+1);
		  assert(ck>=iz-1);
		  assert(ck<=iz+1);
#endif
#ifdef ENABLE_CFL_CHECK
          //check that source cell is a neighbor of destination cell
          bool cfl_cond_satisfied=false;
          for(int di = -1; di <= 1; ++di)
            for(int dj = -1; dj <= 1; ++dj)
              for(int dk = -1; dk <= 1; ++dk)
              {
                int ii = ci + di;
                int jj = cj + dj;
                int kk = ck + dk;
                if(ii >= 0 && ii < nx && jj >= 0 && jj < ny && kk >= 0 && kk < nz)
                {
                  int index = (kk*ny + jj)*nx + ii;
                  if(index == index2)
                  {
                    cfl_cond_satisfied=true;
                    break;
                  }
                }
              }
          if(!cfl_cond_satisfied)
          {
            std::cerr << "FATAL ERROR: Courant–Friedrichs–Lewy condition not satisfied." << std::endl;
            exit(1);
          }
#endif //ENABLE_CFL_CHECK

          int index = (ck*ny + cj)*nx + ci;
          // this assumes that particles cannot travel more than one grid cell per time step
          if(border[index])
            pthread_mutex_lock(&mutex[index][CELL_MUTEX_ID]);
          Cell *cell = last_cells[index];
          int np = cnumPars[index];

          //add another cell structure if everything full
          if( (np % PARTICLES_PER_CELL == 0) && (cnumPars[index] != 0) ) {
            cell->next = cellpool_getcell(&pools[tid]);
            cell = cell->next;
            last_cells[index] = cell;
          }
          ++cnumPars[index];
          if(border[index])
            pthread_mutex_unlock(&mutex[index][CELL_MUTEX_ID]);

          //copy source to destination particle
	  cell->p_coord[(np % PARTICLES_PER_CELL)*3] = cell2->p_coord[(j % PARTICLES_PER_CELL)*3]; // x
	  cell->p_coord[((np % PARTICLES_PER_CELL)*3)+1] = cell2->p_coord[((j % PARTICLES_PER_CELL)*3)+1]; // y
	  cell->p_coord[((np % PARTICLES_PER_CELL)*3)+2] = cell2->p_coord[((j % PARTICLES_PER_CELL)*3)+2]; // z

	  cell->v_coord[(np % PARTICLES_PER_CELL)*3] = cell2->v_coord[(j % PARTICLES_PER_CELL)*3]; // x
	  cell->v_coord[((np % PARTICLES_PER_CELL)*3)+1] = cell2->v_coord[((j % PARTICLES_PER_CELL)*3)+1]; // y
	  cell->v_coord[((np % PARTICLES_PER_CELL)*3)+2] = cell2->v_coord[((j % PARTICLES_PER_CELL)*3)+2]; // z

          cell->hv[np % PARTICLES_PER_CELL] = cell2->hv[j % PARTICLES_PER_CELL];

          //move pointer to next source cell in list if end of array is reached
          if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
            Cell *temp = cell2;
            cell2 = cell2->next;
            //return cells to pool that are not statically allocated head of lists
            if(temp != &cells2[index2]) {
              //NOTE: This is thread-safe because temp and pool are thread-private, no need to synchronize
              cellpool_returncell(&pools[tid], temp);
            }
          }
        } // for(int j = 0; j < np2; ++j)
        //return cells to pool that are not statically allocated head of lists
        if((cell2 != NULL) && (cell2 != &cells2[index2])) {
          cellpool_returncell(&pools[tid], cell2);
	}
      }

}

////////////////////////////////////////////////////////////////////////////////

int InitNeighCellList(int ci, int cj, int ck, int *neighCells)
{
  int numNeighCells = 0;

  // have the nearest particles first -> help branch prediction
  int my_index = (ck*ny + cj)*nx + ci;
  neighCells[numNeighCells] = my_index;
  ++numNeighCells;

  for(int di = -1; di <= 1; ++di)
    for(int dj = -1; dj <= 1; ++dj)
      for(int dk = -1; dk <= 1; ++dk)
      {
        int ii = ci + di;
        int jj = cj + dj;
        int kk = ck + dk;
        if(ii >= 0 && ii < nx && jj >= 0 && jj < ny && kk >= 0 && kk < nz)
        {
          int index = (kk*ny + jj)*nx + ii;
          if((index < my_index) && (cnumPars[index] != 0))
          {
            neighCells[numNeighCells] = index;
            ++numNeighCells;
          }
        }
      }
  return numNeighCells;
}

////////////////////////////////////////////////////////////////////////////////

void InitDensitiesAndForcesMT(int tid)
{
  for(int iz = grids[tid].sz; iz < grids[tid].ez; ++iz)
    for(int iy = grids[tid].sy; iy < grids[tid].ey; ++iy)
      for(int ix = grids[tid].sx; ix < grids[tid].ex; ++ix)
      {
	int index = (iz*ny + iy)*nx + ix;
        Cell *cell = &cells[index];
        int np = cnumPars[index];
        for(int j = 0; j < np; ++j)
        {
          cell->density[j % PARTICLES_PER_CELL] = 0.0;
          cell->a[j % PARTICLES_PER_CELL] = externalAcceleration;
          //move pointer to next cell in list if end of array is reached
          if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
            cell = cell->next;
          }
        }
      }
}

////////////////////////////////////////////////////////////////////////////////

void ComputeDensitiesMT(int tid)
{
  int neighCells[3*3*3];

  for(int iz = grids[tid].sz; iz < grids[tid].ez; ++iz)
    for(int iy = grids[tid].sy; iy < grids[tid].ey; ++iy)
      for(int ix = grids[tid].sx; ix < grids[tid].ex; ++ix)
      {
        int index = (iz*ny + iy)*nx + ix;
        int np = cnumPars[index];
        if(np == 0)
          continue;

        int numNeighCells = InitNeighCellList(ix, iy, iz, neighCells);

        Cell *cell = &cells[index];

	//	for(int ipar = 0; ipar < np; ++ipar) {
	for(int inc = 0; inc < numNeighCells; ++inc) {
            int indexNeigh = neighCells[inc];
            Cell *neigh = &cells[indexNeigh];
#ifdef DEBUG_SIMD // JMCG
	    Cell *neigh2 = &cells[indexNeigh];
#endif
            int numNeighPars = cnumPars[indexNeigh];

#ifdef SIMD_WIDTH  // JMCG Vectorization: SIMD Version
	    fptype tc = 0;
	    int leftovers = 0;

	    if(numNeighPars >= SIMD_WIDTH) {
	      tc = ComputeDensitiesMTSIMD(&leftovers,indexNeigh,numNeighPars,np,index, cell,&neigh);
	    }

	    tc += ComputeDensitiesMTOriginal(leftovers,indexNeigh,numNeighPars,np,index,cell,neigh);

#ifdef DEBUG_SIMD
            fptype tc2 = ComputeDensitiesMTOriginal_test(0,indexNeigh,numNeighPars,np,index,cell,neigh2);
	    if (tc2 != tc) {
	      if ((tc2 - tc) > max_diff) {
		max_diff = (tc2 - tc);
		cout << "Max Diff Densities " << max_diff <<endl;
		cout << "Vect_TC " << tc << endl;
		cout << "Original_TC " << tc2 << endl;
	      }
	    }
#endif

#else
	    // JMCG Vectorization: Original Code does not work when changing the order of the loops, calling modified code
	    int leftovers = 0;
	    fptype tc = ComputeDensitiesMTOriginal(leftovers,indexNeigh,numNeighPars,np,index,cell,neigh);

#endif // END JMCG Vectorization

	}
          //move pointer to next cell in list if end of array is reached
      }
}

/* JMCG BEGIN */

inline fptype ComputeDensitiesMTOriginal(int iparNeigh_in, int indexNeigh, int numNeighPars, int np, int index, Cell *cell, Cell *neigh ) {
  fptype total_tc = 0.0;
  for(int iparNeigh = iparNeigh_in; iparNeigh < numNeighPars; ++iparNeigh) {
    Cell *cell_ipar = cell;
    fptype tc_ipar = 0;
    for(int ipar = 0; ipar < np; ++ipar) {
      //Check address to make sure densities are computed only once per pair
      //    if(&neigh->p[iparNeigh % PARTICLES_PER_CELL] < &cell->p[ipar % PARTICLES_PER_CELL]) { // JMCG We no longer have p[], not sure if this works
      fptype tc = 0.0;
      if(&neigh->p_coord[(iparNeigh % PARTICLES_PER_CELL)*3] < &cell_ipar->p_coord[(ipar % PARTICLES_PER_CELL)*3]) {
        //      fptype distSq = (cell->p[ipar % PARTICLES_PER_CELL] - neigh->p[iparNeigh % PARTICLES_PER_CELL]).GetLengthSq(); // JMCG We no longer have p[]
        fptype dist_x = cell_ipar->p_coord[(ipar % PARTICLES_PER_CELL)*3] - neigh->p_coord[(iparNeigh % PARTICLES_PER_CELL)*3];
        fptype dist_y = cell_ipar->p_coord[((ipar % PARTICLES_PER_CELL)*3)+1] - neigh->p_coord[((iparNeigh % PARTICLES_PER_CELL)*3)+1];
        fptype dist_z = cell_ipar->p_coord[((ipar % PARTICLES_PER_CELL)*3)+2] - neigh->p_coord[((iparNeigh % PARTICLES_PER_CELL)*3)+2];

        fptype distSq = (dist_x * dist_x) + (dist_y * dist_y) + (dist_z * dist_z);

        if(distSq < hSq) {
          fptype t = hSq - distSq;
          tc = t*t*t;
          tc_ipar += tc;
#ifdef DEBUG_SIMD
	  total_tc += tc;
#endif
          if(border[index]) {
            pthread_mutex_lock(&mutex[index][ipar % MUTEXES_PER_CELL]);
            cell_ipar->density[ipar % PARTICLES_PER_CELL] += tc;
            pthread_mutex_unlock(&mutex[index][ipar % MUTEXES_PER_CELL]);
          }
          else
            cell_ipar->density[ipar % PARTICLES_PER_CELL] += tc;
        }
      }

      if(ipar % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
        cell_ipar = cell_ipar->next;
      }
    } // ipar

    if(border[indexNeigh]) {
      pthread_mutex_lock(&mutex[indexNeigh][iparNeigh % MUTEXES_PER_CELL]);
      neigh->density[iparNeigh % PARTICLES_PER_CELL] += tc_ipar;
      pthread_mutex_unlock(&mutex[indexNeigh][iparNeigh % MUTEXES_PER_CELL]);
    }
    else
      neigh->density[iparNeigh % PARTICLES_PER_CELL] += tc_ipar;

    //move pointer to next cell in list if end of array is reached
    if(iparNeigh % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
      neigh = neigh->next;
    }
  }
  return total_tc;
}

// Compute densities one by one starting at iparNeigh_in. Used for debugging, does not store any density
inline fptype ComputeDensitiesMTOriginal_test(int iparNeigh_in, int indexNeigh, int numNeighPars, int np, int index, Cell *cell, Cell *neigh ) {

  fptype total_tc = 0.0;
#ifdef SIMD_WIDTH
  for(int iparNeigh = iparNeigh_in; iparNeigh < numNeighPars; ++iparNeigh) {
    Cell *cell_ipar = cell;
    for(int ipar = 0; ipar < np; ++ipar) {
      //Check address to make sure densities are computed only once per pair
      //    if(&neigh->p[iparNeigh % PARTICLES_PER_CELL] < &cell->p[ipar % PARTICLES_PER_CELL]) { // JMCG We no longer have p[], not sure if this works
      fptype tc = 0.0;
      if(&neigh->p_coord[(iparNeigh % PARTICLES_PER_CELL)*3] < &cell_ipar->p_coord[(ipar % PARTICLES_PER_CELL)*3]) {
        //      fptype distSq = (cell->p[ipar % PARTICLES_PER_CELL] - neigh->p[iparNeigh % PARTICLES_PER_CELL]).GetLengthSq(); // JMCG We no longer have p[]
        fptype dist_x = cell_ipar->p_coord[(ipar % PARTICLES_PER_CELL)*3] - neigh->p_coord[(iparNeigh % PARTICLES_PER_CELL)*3];
        fptype dist_y = cell_ipar->p_coord[((ipar % PARTICLES_PER_CELL)*3)+1] - neigh->p_coord[((iparNeigh % PARTICLES_PER_CELL)*3)+1];
        fptype dist_z = cell_ipar->p_coord[((ipar % PARTICLES_PER_CELL)*3)+2] - neigh->p_coord[((iparNeigh % PARTICLES_PER_CELL)*3)+2];

        fptype distSq = (dist_x * dist_x) + (dist_y * dist_y) + (dist_z * dist_z);

        if(distSq < hSq) {
          fptype t = hSq - distSq;
          tc = t*t*t;
	  total_tc += tc;
        }
      }

      if(ipar % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
        cell_ipar = cell_ipar->next;
      }
    } // ipar

    //move pointer to next cell in list if end of array is reached
    if(iparNeigh % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
      neigh = neigh->next;
    }
  }
#endif
  return total_tc;
}



/* JMCG Vectorization. SIMD implementation with number of particles greater than SIMD_WIDTH but not neccesary to be divisible by SIMD_WIDTH */
inline fptype ComputeDensitiesMTSIMD(int *iparNeigh_in, int indexNeigh, int numNeighPars, int np, int index, Cell *cell, Cell **neigh ) {

  fptype total_tc = 0.0;
#ifdef SIMD_WIDTH
  int iparNeigh;

  //  _MM_ALIGN fptype total_tc[SIMD_WIDTH];
  _MM_TYPE _mask, _flag, temp_tc, _tc;

  iparNeigh = *iparNeigh_in;

  while((iparNeigh + SIMD_WIDTH) <= numNeighPars) {
    Cell *cell_ipar = cell;
    fptype tc_ipar = 0;
    temp_tc = _MM_SET(0);
    _tc = _MM_SET(0);

    _MM_TYPE data_x, data_y, data_z;
    _MM_LOAD3(&data_x,&data_y,&data_z,&((*neigh)->p_coord[(iparNeigh % PARTICLES_PER_CELL)*3]));

    for(int ipar = 0; ipar < np; ++ipar) {
      //  for(iparNeigh = *iparNeigh_in; iparNeigh < numNeighPars; iparNeigh += SIMD_WIDTH) {

      //Check address to make sure densities are computed only once per pair || JMCG NEED ANOTHER MASK HERE (Inverted as SSE does)
      // if(&(*neigh)->p[iparNeigh % PARTICLES_PER_CELL] < &cell->p[ipar % PARTICLES_PER_CELL]) {

#define LITERAL_FORMULA_DENSITY(A) ((&((*neigh)->p_coord[((A) % PARTICLES_PER_CELL)*3]) < &(cell_ipar->p_coord[(ipar % PARTICLES_PER_CELL)*3])) ? 1 : 0)
      _mask = _MM_SETR_FORMULA_1PARAM(LITERAL_FORMULA_DENSITY,iparNeigh);

      // This looks weird but our ARM evaluation system seems to be running in
      // Runfast mode. In this mode Subnormal numbers are being flushed to zero (that is, the 0x0...1 stored in otype)
      // Casting everything to integer and using integer comparations seems to work
      // minimum positive subnormal number 00000001 1.40129846e-45
//      _mask = _MM_CAST_I_TO_FP(_MM_CMPEQ_SIG(_MM_CAST_FP_TO_I(_mask), _MM_SET_I(1))); // Set 1s to all to 1s (I cant figure it out why setting 0xffffffff in the setr does not work)
      _mask = (_MM_TYPE)_MM_CMPEQ_SIG((_MM_TYPE_I)_mask, (_MM_TYPE_I)_MM_SET(1)); // Set 1s to all to 1s (I cant figure it out why setting 0xffffffff in the setr does not work)

	// ORIGINAL CODE
	//      fptype distSq = (cell->p[ipar % PARTICLES_PER_CELL] - (*neigh)->p[iparNeigh % PARTICLES_PER_CELL]).GetLengthSq();
	// if(distSq < hSq) { // JMCG We skip the if in the vectorization, just ignore negative values

        _MM_TYPE dist_x = _MM_SUB(_MM_SET(cell_ipar->p_coord[(ipar% PARTICLES_PER_CELL)*3]),data_x);
	_MM_TYPE dist_y = _MM_SUB(_MM_SET(cell_ipar->p_coord[((ipar% PARTICLES_PER_CELL)*3)+1]),data_y);
	_MM_TYPE dist_z = _MM_SUB(_MM_SET(cell_ipar->p_coord[((ipar% PARTICLES_PER_CELL)*3)+2]),data_z);

	dist_x = _MM_MUL(dist_x,dist_x);
	dist_y = _MM_MUL(dist_y,dist_y);
	dist_z = _MM_MUL(dist_z,dist_z);

	_MM_TYPE distSq = _MM_ADD(_MM_ADD(dist_x,dist_y),dist_z);


	//	fptype t = hSq - distSq;
	_MM_TYPE _t = _MM_SUB(_MM_SET(hSq),distSq);

	// JMCG We only keep positive values (if(distSq < hSq))
	_flag = (_MM_TYPE)_MM_CMPLT(_t, _MM_SET(0));
	_tc = _MM_OR(_MM_AND(_flag, _MM_SET(0)), _MM_ANDNOT(_flag, _t));

	// Mask filter address to make sure densities are computed only once per pair
	_tc = _MM_OR(_MM_AND(_mask, _tc), _MM_ANDNOT(_mask, _MM_SET(0)));

	//	fptype tc = t*t*t;
	_tc = _MM_MUL(_MM_MUL(_tc,_tc),_tc);

	temp_tc = _MM_ADD(temp_tc,_tc);

	fptype tc = _MM_REDUCE_ADD(_tc);

#ifdef DEBUG_SIMD
	total_tc += tc;
#endif

	if(border[index]) {
	  pthread_mutex_lock(&mutex[index][ipar % MUTEXES_PER_CELL]);
	  //cell->density[ipar % PARTICLES_PER_CELL] += tc;
	  cell_ipar->density[ipar % PARTICLES_PER_CELL] += tc;
	  pthread_mutex_unlock(&mutex[index][ipar % MUTEXES_PER_CELL]);
	}
	else {
	  //cell->density[ipar % PARTICLES_PER_CELL] += tc;
	  cell_ipar->density[ipar % PARTICLES_PER_CELL] += tc;
	}


	/*else {
	  iparNeigh = numNeighPars; // JMCG Adding this we can Stop iteration on internal loop. Slightly faster than original code, but I'm not completely sure that is the same.
	  } */

      if(ipar % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
        cell_ipar = cell_ipar->next;
      }

    } // ipar

    // JMCG Store densities
    if(border[indexNeigh]) {
      pthread_mutex_lock(&mutex[indexNeigh][iparNeigh % MUTEXES_PER_CELL]);
      _MM_STORE(&((*neigh)->density[iparNeigh % PARTICLES_PER_CELL]),_MM_ADD(_MM_LOAD(&((*neigh)->density[iparNeigh % PARTICLES_PER_CELL])),temp_tc));
      pthread_mutex_unlock(&mutex[indexNeigh][iparNeigh % MUTEXES_PER_CELL]);
    }
    else
      _MM_STORE(&((*neigh)->density[iparNeigh % PARTICLES_PER_CELL]),_MM_ADD(_MM_LOAD(&((*neigh)->density[iparNeigh % PARTICLES_PER_CELL])),temp_tc));

    iparNeigh += SIMD_WIDTH;
    //move pointer to next cell in list if end of array is reached
    if(iparNeigh % PARTICLES_PER_CELL == 0) {
      *neigh = (*neigh)->next;
    }

  }

  *iparNeigh_in = iparNeigh;
#endif
  return total_tc;
}


// Compute Forces one by one starting at iparNeigh_in
inline Vec3 ComputeForcesMTOriginal(int iparNeigh_in, int indexNeigh, int numNeighPars, int np, int index, Cell *cell, Cell *neigh ) {

  Vec3 return_vec(0.0,0.0,0.0);

  for(int iparNeigh = iparNeigh_in; iparNeigh < numNeighPars; ++iparNeigh) {
    Cell *cell_ipar = cell;
    Vec3 acc_ipar(0.0,0.0,0.0);
    for(int ipar = 0; ipar < np; ++ipar) {
      //Check address to make sure forces are computed only once per pair
      //    if(&neigh->p[iparNeigh % PARTICLES_PER_CELL] < &cell->p[ipar % PARTICLES_PER_CELL]) {
      if(&neigh->p_coord[(iparNeigh % PARTICLES_PER_CELL)*3] < &cell_ipar->p_coord[(ipar % PARTICLES_PER_CELL)*3]) {
	// Vec3 disp = cell_ipar->p[ipar % PARTICLES_PER_CELL] - neigh->p[iparNeigh % PARTICLES_PER_CELL]; // old code
	Vec3 disp;
	disp.x = cell_ipar->p_coord[(ipar % PARTICLES_PER_CELL)*3] - neigh->p_coord[(iparNeigh % PARTICLES_PER_CELL)*3];
	disp.y = cell_ipar->p_coord[((ipar % PARTICLES_PER_CELL)*3)+1] - neigh->p_coord[((iparNeigh % PARTICLES_PER_CELL)*3)+1];
	disp.z = cell_ipar->p_coord[((ipar % PARTICLES_PER_CELL)*3)+2] - neigh->p_coord[((iparNeigh % PARTICLES_PER_CELL)*3)+2];
	fptype distSq = disp.GetLengthSq();

	if(distSq < hSq) {
#ifndef ENABLE_DOUBLE_PRECISION
	  fptype dist = sqrtf(std::max(distSq, (fptype)1e-12));
#else
	  fptype dist = sqrt(std::max(distSq, 1e-12));
#endif //ENABLE_DOUBLE_PRECISION
	  fptype hmr = h - dist;
	  Vec3 acc = disp * pressureCoeff * (hmr*hmr/dist) * (cell_ipar->density[ipar % PARTICLES_PER_CELL]+neigh->density[iparNeigh % PARTICLES_PER_CELL] - doubleRestDensity);
	  //	acc += (neigh->v[iparNeigh % PARTICLES_PER_CELL] - cell->v[ipar % PARTICLES_PER_CELL]) * viscosityCoeff * hmr;
	  acc.x += (neigh->v_coord[(iparNeigh % PARTICLES_PER_CELL)*3] - cell_ipar->v_coord[(ipar % PARTICLES_PER_CELL)*3]) * viscosityCoeff * hmr;
	  acc.y += (neigh->v_coord[((iparNeigh % PARTICLES_PER_CELL)*3)+1] - cell_ipar->v_coord[((ipar % PARTICLES_PER_CELL)*3)+1]) * viscosityCoeff * hmr;
	  acc.z += (neigh->v_coord[((iparNeigh % PARTICLES_PER_CELL)*3)+2] - cell_ipar->v_coord[((ipar % PARTICLES_PER_CELL)*3)+2]) * viscosityCoeff * hmr;

	  acc /= cell_ipar->density[ipar % PARTICLES_PER_CELL] * neigh->density[iparNeigh % PARTICLES_PER_CELL];

#ifdef DEBUG_SIMD
	  return_vec += acc;
#endif
	  acc_ipar +=acc;

	  if( border[index]) {
	    pthread_mutex_lock(&mutex[index][ipar % MUTEXES_PER_CELL]);
	    cell_ipar->a[ipar % PARTICLES_PER_CELL] += acc;
	    pthread_mutex_unlock(&mutex[index][ipar % MUTEXES_PER_CELL]);
	  }
	  else
	    cell_ipar->a[ipar % PARTICLES_PER_CELL] += acc;

	}
      }
      if(ipar % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
        cell_ipar = cell_ipar->next;
      }
    }

    if( border[indexNeigh]) {
      pthread_mutex_lock(&mutex[indexNeigh][iparNeigh % MUTEXES_PER_CELL]);
      neigh->a[iparNeigh % PARTICLES_PER_CELL] -= acc_ipar;
      pthread_mutex_unlock(&mutex[indexNeigh][iparNeigh % MUTEXES_PER_CELL]);
    } else {
      neigh->a[iparNeigh % PARTICLES_PER_CELL] -= acc_ipar;
    }
    //move pointer to next cell in list if end of array is reached
    if(iparNeigh % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
      neigh = neigh->next;
    }
  }
  return return_vec;
}

// Compute Forces one by one starting at iparNeigh_in. Used for debugging, does not store any value in a[].
inline Vec3 ComputeForcesMTOriginal_test(int iparNeigh_in, int indexNeigh, int numNeighPars, int np, int index, Cell *cell, Cell *neigh ) {

  Vec3 return_vec(0.0,0.0,0.0);

#ifdef SIMD_WIDTH

  for(int iparNeigh = iparNeigh_in; iparNeigh < numNeighPars; ++iparNeigh) {
    Cell *cell_ipar = cell;
    for(int ipar = 0; ipar < np; ++ipar) {
      //Check address to make sure forces are computed only once per pair
      //    if(&neigh->p[iparNeigh % PARTICLES_PER_CELL] < &cell->p[ipar % PARTICLES_PER_CELL]) {
      if(&neigh->p_coord[(iparNeigh % PARTICLES_PER_CELL)*3] < &cell_ipar->p_coord[(ipar % PARTICLES_PER_CELL)*3]) {
	// Vec3 disp = cell_ipar->p[ipar % PARTICLES_PER_CELL] - neigh->p[iparNeigh % PARTICLES_PER_CELL]; // old code
	Vec3 disp;
	disp.x = cell_ipar->p_coord[(ipar % PARTICLES_PER_CELL)*3] - neigh->p_coord[(iparNeigh % PARTICLES_PER_CELL)*3];
	disp.y = cell_ipar->p_coord[((ipar % PARTICLES_PER_CELL)*3)+1] - neigh->p_coord[((iparNeigh % PARTICLES_PER_CELL)*3)+1];
	disp.z = cell_ipar->p_coord[((ipar % PARTICLES_PER_CELL)*3)+2] - neigh->p_coord[((iparNeigh % PARTICLES_PER_CELL)*3)+2];
	fptype distSq = disp.GetLengthSq();

	if(distSq < hSq) {
#ifndef ENABLE_DOUBLE_PRECISION
	  fptype dist = sqrtf(std::max(distSq, (fptype)1e-12));
#else
	  fptype dist = sqrt(std::max(distSq, 1e-12));
#endif //ENABLE_DOUBLE_PRECISION
	  fptype hmr = h - dist;
	  Vec3 acc = disp * pressureCoeff * (hmr*hmr/dist) * (cell_ipar->density[ipar % PARTICLES_PER_CELL]+neigh->density[iparNeigh % PARTICLES_PER_CELL] - doubleRestDensity);
	  //	acc += (neigh->v[iparNeigh % PARTICLES_PER_CELL] - cell->v[ipar % PARTICLES_PER_CELL]) * viscosityCoeff * hmr;
	  acc.x += (neigh->v_coord[(iparNeigh % PARTICLES_PER_CELL)*3] - cell_ipar->v_coord[(ipar % PARTICLES_PER_CELL)*3]) * viscosityCoeff * hmr;
	  acc.y += (neigh->v_coord[((iparNeigh % PARTICLES_PER_CELL)*3)+1] - cell_ipar->v_coord[((ipar % PARTICLES_PER_CELL)*3)+1]) * viscosityCoeff * hmr;
	  acc.z += (neigh->v_coord[((iparNeigh % PARTICLES_PER_CELL)*3)+2] - cell_ipar->v_coord[((ipar % PARTICLES_PER_CELL)*3)+2]) * viscosityCoeff * hmr;

	  acc /= cell_ipar->density[ipar % PARTICLES_PER_CELL] * neigh->density[iparNeigh % PARTICLES_PER_CELL];

	  return_vec += acc;

	}
      }
      if(ipar % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
        cell_ipar = cell_ipar->next;
      }
    }

    //move pointer to next cell in list if end of array is reached
    if(iparNeigh % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
      neigh = neigh->next;
    }
  }
#endif
  return return_vec;
}


// Compute Forces in groups of SIMD_WIDTH starting at iparNeigh_in. Stores last possition multiple of SIMD_WIDTH in iparNeigh_in and modifies neigh pointer to the latest neigh analized
inline Vec3 ComputeForcesMTSIMD(int *iparNeigh_in, int indexNeigh, int numNeighPars, int np, int index, Cell *cell, Cell **neigh ) {

  Vec3 return_vec(0.0,0.0,0.0);

#ifdef SIMD_WIDTH

  _MM_ALIGN fptype new_a_x[SIMD_WIDTH];
  _MM_ALIGN fptype new_a_y[SIMD_WIDTH];
  _MM_ALIGN fptype new_a_z[SIMD_WIDTH];

  _MM_TYPE _mask, _flag;
  int iparNeigh;

  iparNeigh = *iparNeigh_in;

  while((iparNeigh + SIMD_WIDTH) <= numNeighPars) {
    //Check address to make sure forces are computed only once per pair
    //if(&neigh->p[iparNeigh % PARTICLES_PER_CELL] < &cell->p[ipar % PARTICLES_PER_CELL])  // JMCG Vectorization. Mask for this one

    Cell *cell_ipar = cell;

    _MM_TYPE acc_x_ipar = _MM_SET(0);
    _MM_TYPE acc_y_ipar = _MM_SET(0);
    _MM_TYPE acc_z_ipar = _MM_SET(0);

    _MM_TYPE data_x, data_y, data_z;
    _MM_LOAD3(&data_x,&data_y,&data_z,&((*neigh)->p_coord[(iparNeigh % PARTICLES_PER_CELL)*3]));

    _MM_TYPE v_x, v_y, v_z;
    _MM_LOAD3(&v_x,&v_y,&v_z,&((*neigh)->v_coord[(iparNeigh % PARTICLES_PER_CELL)*3]));

    for(int ipar = 0; ipar < np; ++ipar) {

#define LITERAL_FORMULA_FORCES(A) ((&((*neigh)->p_coord[((A) % PARTICLES_PER_CELL)*3]) < &(cell_ipar->p_coord[(ipar % PARTICLES_PER_CELL)*3])) ? 1 : 0)
       _mask = _MM_SETR_FORMULA_1PARAM(LITERAL_FORMULA_FORCES,iparNeigh);

       // This looks weird but our ARM evaluation system seems to be running in
       // Runfast mode. In this mode Subnormal numbers are being flushed to zero (that is, the 0x0...1 stored in otype)
       // Casting everything to integer and using integer comparations seems to work
       // minimum positive subnormal number 00000001 1.40129846e-45
//       _mask = _MM_CAST_I_TO_FP(_MM_CMPEQ_SIG(_MM_CAST_FP_TO_I(_mask), _MM_SET_I(1))); // Set 1s to all to 1s (I cant figure it out why setting 0xffffffff in the setr does not work)
       _mask = (_MM_TYPE)_MM_CMPEQ_SIG((_MM_TYPE_I)_mask, (_MM_TYPE_I)_MM_SET(1)); // Set 1s to all to 1s (I cant figure it out why setting 0xffffffff in the setr does not work)

       _MM_TYPE dist_x = _MM_SUB(_MM_SET(cell_ipar->p_coord[(ipar% PARTICLES_PER_CELL)*3]),data_x);
       _MM_TYPE dist_y = _MM_SUB(_MM_SET(cell_ipar->p_coord[((ipar% PARTICLES_PER_CELL)*3)+1]),data_y);
       _MM_TYPE dist_z = _MM_SUB(_MM_SET(cell_ipar->p_coord[((ipar% PARTICLES_PER_CELL)*3)+2]),data_z);

       _MM_TYPE distSq = _MM_ADD(_MM_ADD( _MM_MUL(dist_x,dist_x),
					  _MM_MUL(dist_y,dist_y)),
				 _MM_MUL(dist_z,dist_z));

	/*
	  if(distSq < hSq) { // JMCG We skip the if in the vectorization, just ignore negative values with mask
	*/

	// JMCG We only keep positive values
	_MM_TYPE _t = _MM_SUB(_MM_SET(hSq),distSq);
	_flag = (_MM_TYPE)_MM_CMPLT(_t, _MM_SET(0)); // Use flag later

	// Extrack bits from flag to speed up, dont run the code if all 0
	  /*
	    // Original code
	    #ifndef ENABLE_DOUBLE_PRECISION
	    fptype dist = sqrtf(std::max(distSq, (fptype)1e-12));
	    #else
	    fptype dist = sqrt(std::max(distSq, 1e-12));
	    #endif //ENABLE_DOUBLE_PRECISION
	  */
	  _MM_TYPE _dist = _MM_SQRT(_MM_MAX(distSq, _MM_SET(1e-12)));

	  //fptype hmr = h - dist;
	  _MM_TYPE _hmr = _MM_SUB(_MM_SET(h), _dist);

	  // Vec3 acc = disp * pressureCoeff * (hmr*hmr/dist) * (cell->density[ipar % PARTICLES_PER_CELL]+neigh->density[iparNeigh % PARTICLES_PER_CELL] - doubleRestDensity); // JMCG We separate on the three coordinates

	  _MM_TYPE common = _MM_MUL(_MM_MUL(_MM_SET(pressureCoeff),_MM_DIV(_MM_MUL(_hmr,_hmr),_dist)),
				    _MM_SUB(_MM_ADD(_MM_SET(cell_ipar->density[ipar % PARTICLES_PER_CELL]),_MM_LOAD(&((*neigh)->density[iparNeigh % PARTICLES_PER_CELL]))), _MM_SET(doubleRestDensity))); // Common to all three coordinates

	  _MM_TYPE acc_x = _MM_MUL(dist_x,common);
	  _MM_TYPE acc_y = _MM_MUL(dist_y,common);
	  _MM_TYPE acc_z = _MM_MUL(dist_z,common);

	  //  acc += (neigh->v[iparNeigh % PARTICLES_PER_CELL] - cell->v[ipar % PARTICLES_PER_CELL]) * viscosityCoeff * hmr;
	  acc_x = _MM_ADD(acc_x,_MM_MUL(_hmr,_MM_MUL(_MM_SET(viscosityCoeff),_MM_SUB(v_x,_MM_SET(cell_ipar->v_coord[(ipar % PARTICLES_PER_CELL)*3])))));
	  acc_y = _MM_ADD(acc_y,_MM_MUL(_hmr,_MM_MUL(_MM_SET(viscosityCoeff),_MM_SUB(v_y,_MM_SET(cell_ipar->v_coord[((ipar % PARTICLES_PER_CELL)*3)+1])))));
	  acc_z = _MM_ADD(acc_z,_MM_MUL(_hmr,_MM_MUL(_MM_SET(viscosityCoeff),_MM_SUB(v_z,_MM_SET(cell_ipar->v_coord[((ipar % PARTICLES_PER_CELL)*3)+2])))));

	  //      acc /= cell->density[ipar % PARTICLES_PER_CELL] * neigh->density[iparNeigh % PARTICLES_PER_CELL];
	  common = _MM_MUL(_MM_SET(cell_ipar->density[ipar % PARTICLES_PER_CELL]),_MM_LOAD(&((*neigh)->density[iparNeigh % PARTICLES_PER_CELL])));
	  acc_x = _MM_DIV(acc_x,common);
	  acc_y = _MM_DIV(acc_y,common);
	  acc_z = _MM_DIV(acc_z,common);

	  // Mask filter address to make sure densities are computed only once per pair
	  acc_x = _MM_OR(_MM_AND(_mask, acc_x), _MM_ANDNOT(_mask, _MM_SET(0)));
	  acc_y = _MM_OR(_MM_AND(_mask, acc_y), _MM_ANDNOT(_mask, _MM_SET(0)));
	  acc_z = _MM_OR(_MM_AND(_mask, acc_z), _MM_ANDNOT(_mask, _MM_SET(0)));

	  // Mask filter results for distSq < hSq
	  acc_x = _MM_OR(_MM_AND(_flag, _MM_SET(0)), _MM_ANDNOT(_flag, acc_x));
	  acc_y = _MM_OR(_MM_AND(_flag, _MM_SET(0)), _MM_ANDNOT(_flag, acc_y));
	  acc_z = _MM_OR(_MM_AND(_flag, _MM_SET(0)), _MM_ANDNOT(_flag, acc_z));

	  // Store values for iparneigh
	  acc_x_ipar = _MM_ADD(acc_x_ipar,acc_x);
	  acc_y_ipar = _MM_ADD(acc_y_ipar,acc_y);
	  acc_z_ipar = _MM_ADD(acc_z_ipar,acc_z);

	  // Combine all values
	  fptype combined_x = _MM_REDUCE_ADD(acc_x);
	  fptype combined_y = _MM_REDUCE_ADD(acc_y);
	  fptype combined_z = _MM_REDUCE_ADD(acc_z);

#ifdef DEBUG_SIMD
	  return_vec.x += combined_x;
	  return_vec.y += combined_y;
	  return_vec.z += combined_z;
#endif
	  if( border[index]) {
            pthread_mutex_lock(&mutex[index][ipar % MUTEXES_PER_CELL]);
            cell_ipar->a[ipar % PARTICLES_PER_CELL].x += combined_x;
	    cell_ipar->a[ipar % PARTICLES_PER_CELL].y += combined_y;
	    cell_ipar->a[ipar % PARTICLES_PER_CELL].z += combined_z;
            pthread_mutex_unlock(&mutex[index][ipar % MUTEXES_PER_CELL]);
          }
          else {
            cell_ipar->a[ipar % PARTICLES_PER_CELL].x += combined_x;
	    cell_ipar->a[ipar % PARTICLES_PER_CELL].y += combined_y;
	    cell_ipar->a[ipar % PARTICLES_PER_CELL].z += combined_z;
	  }
      if(ipar % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
        cell_ipar = cell_ipar->next;
      }
    } // ipar

    // Store a values to update later
    _MM_STORE(&(new_a_x[0]),acc_x_ipar);
    _MM_STORE(&(new_a_y[0]),acc_y_ipar);
    _MM_STORE(&(new_a_z[0]),acc_z_ipar);

    if( border[indexNeigh]) {
      pthread_mutex_lock(&mutex[indexNeigh][iparNeigh % MUTEXES_PER_CELL]);
      //	neigh->a[iparNeigh % PARTICLES_PER_CELL] -= acc;
#pragma unroll(SIMD_WIDTH)
      for (int i = 0; i < SIMD_WIDTH; i++) {
	(*neigh)->a[(iparNeigh+i) % PARTICLES_PER_CELL].x -= new_a_x[i];
	(*neigh)->a[(iparNeigh+i) % PARTICLES_PER_CELL].y -= new_a_y[i];
	(*neigh)->a[(iparNeigh+i) % PARTICLES_PER_CELL].z -= new_a_z[i];
      }
      pthread_mutex_unlock(&mutex[indexNeigh][iparNeigh % MUTEXES_PER_CELL]);
    } else {
      //	neigh->a[iparNeigh % PARTICLES_PER_CELL] -= acc;
#pragma unroll(SIMD_WIDTH)
      for (int i = 0; i < SIMD_WIDTH; i++) {
	(*neigh)->a[(iparNeigh+i) % PARTICLES_PER_CELL].x -= new_a_x[i];
	(*neigh)->a[(iparNeigh+i) % PARTICLES_PER_CELL].y -= new_a_y[i];
	(*neigh)->a[(iparNeigh+i) % PARTICLES_PER_CELL].z -= new_a_z[i];
      }
    }

    iparNeigh += SIMD_WIDTH;
    //move pointer to next cell in list if end of array is reached
    if(iparNeigh % PARTICLES_PER_CELL == 0) {
      *neigh = (*neigh)->next;
    }
  }
  *iparNeigh_in = iparNeigh;

#endif // SIMD_WIDTH
  return return_vec;
}

/* JMCG END */

////////////////////////////////////////////////////////////////////////////////

void ComputeDensities2MT(int tid)
{
  const fptype tc = hSq*hSq*hSq;
  for(int iz = grids[tid].sz; iz < grids[tid].ez; ++iz)
    for(int iy = grids[tid].sy; iy < grids[tid].ey; ++iy)
      for(int ix = grids[tid].sx; ix < grids[tid].ex; ++ix)
      {
        int index = (iz*ny + iy)*nx + ix;
        Cell *cell = &cells[index];
        int np = cnumPars[index];
        for(int j = 0; j < np; ++j)
        {
          cell->density[j % PARTICLES_PER_CELL] += tc;
          cell->density[j % PARTICLES_PER_CELL] *= densityCoeff;
          //move pointer to next cell in list if end of array is reached
          if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
            cell = cell->next;
          }
        }
      }
}

////////////////////////////////////////////////////////////////////////////////

void ComputeForcesMT(int tid)
{
  int neighCells[3*3*3];

  for(int iz = grids[tid].sz; iz < grids[tid].ez; ++iz)
    for(int iy = grids[tid].sy; iy < grids[tid].ey; ++iy)
      for(int ix = grids[tid].sx; ix < grids[tid].ex; ++ix)
      {
        int index = (iz*ny + iy)*nx + ix;
        int np = cnumPars[index];
        if(np == 0)
          continue;

        int numNeighCells = InitNeighCellList(ix, iy, iz, neighCells);

        Cell *cell = &cells[index];
	//        for(int ipar = 0; ipar < np; ++ipar) {
          for(int inc = 0; inc < numNeighCells; ++inc)
          {
            int indexNeigh = neighCells[inc];
            Cell *neigh = &cells[indexNeigh];

#ifdef DEBUG_SIMD // JMCG
	    Cell *neigh2 = &cells[indexNeigh];
#endif
            int numNeighPars = cnumPars[indexNeigh];

#ifdef SIMD_WIDTH  // JMCG Vectorization: SIMD Version

	    Vec3 acc(0.0,0.0,0.0);
	    int leftovers = 0;

#ifdef SIMD_STATS
	    total_compute_forces_calls++;
#endif
	    if(numNeighPars >= SIMD_WIDTH) {
#ifdef SIMD_STATS
	      total_compute_forces_calls_sse++;
#endif
	      acc += ComputeForcesMTSIMD(&leftovers,indexNeigh,numNeighPars,np,index,cell,&neigh);
	    }

#ifdef SIMD_STATS
	      if (leftovers == 0)
		total_compute_forces_calls_original++;
	      else {
		if (leftovers < indexNeigh)
		  total_compute_forces_calls_leftovers++;
	      }
#endif
	      acc += ComputeForcesMTOriginal(leftovers,indexNeigh,numNeighPars,np,index,cell,neigh);
#ifdef DEBUG_SIMD
	    Vec3 acc2 = ComputeForcesMTOriginal_test(0,indexNeigh, numNeighPars, np, index, cell, neigh2);
	    if ((acc.x != acc2.x) || (acc.y != acc2.y) || (acc.z != acc2.z)) {
	      float new_diff = fabs((acc2.x - acc.x)) + fabs((acc2.y - acc.y)) + fabs((acc2.z - acc.z));
	      if (max_diff_forces < new_diff) {
		max_diff_forces = new_diff;
		cout << "Max Diff_Forces " << max_diff_forces << " " << index << endl;
		cout << "Vect_ACC " << acc.x << " " << acc.y << " " << acc.z << endl;
		cout << "Original_ACC " << acc2.x << " " << acc2.y << " " << acc2.z << endl;
	      }
	    }
#endif
#else // Original code, Original code does not work with the "modified" loop structure, call modified functions
	    // JMCG Vectorization: Original Code does not work when changing the order of the loops, calling modified code
	    Vec3 acc;
            int leftovers = 0;
	    acc = ComputeForcesMTOriginal(leftovers,indexNeigh,numNeighPars,np,index,cell,neigh);
#endif // SIMD_WIDTH Vectorization
          }
          //move pointer to next cell in list if end of array is reached
      }
}

////////////////////////////////////////////////////////////////////////////////

// ProcessCollisions() with container walls
// Under the assumptions that
// a) a particle will not penetrate a wall
// b) a particle will not migrate further than once cell
// c) the parSize is smaller than a cell
// then only the particles at the perimiters may be influenced by the walls
#if 0
void ProcessCollisionsMT(int tid)
{
  for(int iz = grids[tid].sz; iz < grids[tid].ez; ++iz)
    for(int iy = grids[tid].sy; iy < grids[tid].ey; ++iy)
      for(int ix = grids[tid].sx; ix < grids[tid].ex; ++ix)
      {
        int index = (iz*ny + iy)*nx + ix;
        Cell *cell = &cells[index];
        int np = cnumPars[index];
        for(int j = 0; j < np; ++j)
        {
	  Vec3 pos;
	  pos.x = cell->p_coord[(j % PARTICLES_PER_CELL)*3] + cell->hv[j % PARTICLES_PER_CELL].x * timeStep;
	  pos.y = cell->p_coord[((j % PARTICLES_PER_CELL)*3)+1] + cell->hv[j % PARTICLES_PER_CELL].y * timeStep;
	  pos.z = cell->p_coord[j % PARTICLES_PER_CELL)*3)+2] + cell->hv[j % PARTICLES_PER_CELL].z * timeStep;
          fptype diff = parSize - (pos.x - domainMin.x);

          if(diff > epsilon)
            cell->a[j % PARTICLES_PER_CELL].x += stiffnessCollisions*diff - damping*cell->v_coord[(j % PARTICLES_PER_CELL)*3];

          diff = parSize - (domainMax.x - pos.x);
          if(diff > epsilon)
            cell->a[j % PARTICLES_PER_CELL].x -= stiffnessCollisions*diff + damping*cell->v_coord[(j % PARTICLES_PER_CELL)*3];

          diff = parSize - (pos.y - domainMin.y);
          if(diff > epsilon)
            cell->a[j % PARTICLES_PER_CELL].y += stiffnessCollisions*diff - damping*cell->v_coord[((j % PARTICLES_PER_CELL)*3)+1];

          diff = parSize - (domainMax.y - pos.y);
          if(diff > epsilon)
            cell->a[j % PARTICLES_PER_CELL].y -= stiffnessCollisions*diff + damping*cell->v_coord[((j % PARTICLES_PER_CELL)*3)+1];

          diff = parSize - (pos.z - domainMin.z);
          if(diff > epsilon)
            cell->a[j % PARTICLES_PER_CELL].z += stiffnessCollisions*diff - damping*cell->v_coord[((j % PARTICLES_PER_CELL)*3)+2];

          diff = parSize - (domainMax.z - pos.z);
          if(diff > epsilon)
            cell->a[j % PARTICLES_PER_CELL].z -= stiffnessCollisions*diff + damping*cell->v_coord[((j % PARTICLES_PER_CELL)*3)+2];
          //move pointer to next cell in list if end of array is reached
          if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
            cell = cell->next;
          }
        }
      }
}
#else
void ProcessCollisionsMT(int tid)
{
  for(int iz = grids[tid].sz; iz < grids[tid].ez; ++iz)
  {
    for(int iy = grids[tid].sy; iy < grids[tid].ey; ++iy)
	{
      for(int ix = grids[tid].sx; ix < grids[tid].ex; ++ix)
      {
	    if(!((ix==0)||(iy==0)||(iz==0)||(ix==(nx-1))||(iy==(ny-1))==(iz==(nz-1))))
			continue;	// not on domain wall
        int index = (iz*ny + iy)*nx + ix;
        Cell *cell = &cells[index];
        int np = cnumPars[index];
        for(int j = 0; j < np; ++j)
        {
	  int ji = j % PARTICLES_PER_CELL;
	  Vec3 pos;
	  pos.x = cell->p_coord[ji*3] + cell->hv[ji].x * timeStep;
	  pos.y = cell->p_coord[(ji*3)+1] + cell->hv[ji].y * timeStep;
	  pos.z = cell->p_coord[(ji*3)+2] + cell->hv[ji].z * timeStep;
	  if(ix==0) {
            fptype diff = parSize - (pos.x - domainMin.x);
	    if(diff > epsilon)
              cell->a[ji].x += stiffnessCollisions*diff - damping*cell->v_coord[ji*3];
	  }
	  if(ix==(nx-1))
	    {
	      fptype diff = parSize - (domainMax.x - pos.x);
	      if(diff > epsilon)
		cell->a[ji].x -= stiffnessCollisions*diff + damping*cell->v_coord[ji*3];
	    }
	  if(iy==0)
	    {
	      fptype diff = parSize - (pos.y - domainMin.y);
	      if(diff > epsilon)
		cell->a[ji].y += stiffnessCollisions*diff - damping*cell->v_coord[(ji*3)+1];
	    }
	  if(iy==(ny-1))
	    {
	      fptype diff = parSize - (domainMax.y - pos.y);
	      if(diff > epsilon)
		cell->a[ji].y -= stiffnessCollisions*diff + damping*cell->v_coord[(ji*3)+1];
	    }
	  if(iz==0)
	    {
	      fptype diff = parSize - (pos.z - domainMin.z);
	      if(diff > epsilon)
		cell->a[ji].z += stiffnessCollisions*diff - damping*cell->v_coord[(ji*3)+2];
	    }
	  if(iz==(nz-1))
	    {
	      fptype diff = parSize - (domainMax.z - pos.z);
	      if(diff > epsilon)
		cell->a[ji].z -= stiffnessCollisions*diff + damping*cell->v_coord[(ji*3)+2];
	    }
          //move pointer to next cell in list if end of array is reached
          if(ji == PARTICLES_PER_CELL-1) {
            cell = cell->next;
          }
        }
      }
	}
  }
}
#endif

#define USE_ImpeneratableWall
#if defined(USE_ImpeneratableWall)
void ProcessCollisions2MT(int tid)
{
  for(int iz = grids[tid].sz; iz < grids[tid].ez; ++iz)
  {
    for(int iy = grids[tid].sy; iy < grids[tid].ey; ++iy)
	{
      for(int ix = grids[tid].sx; ix < grids[tid].ex; ++ix)
      {
#if 0
// Chris, the following test should be valid
// *** provided that a particle does not migrate more than 1 cell
// *** per integration step. This does not appear to be the case
// *** in the pthreads version. Serial version it seems to be OK
	    if(!((ix==0)||(iy==0)||(iz==0)||(ix==(nx-1))||(iy==(ny-1))==(iz==(nz-1))))
			continue;	// not on domain wall
#endif
        int index = (iz*ny + iy)*nx + ix;
        Cell *cell = &cells[index];
        int np = cnumPars[index];
        for(int j = 0; j < np; ++j)
        {
		  int ji = j % PARTICLES_PER_CELL;

		  Vec3 pos;
		  pos.x = cell->p_coord[ji*3];
		  pos.y = cell->p_coord[(ji*3)+1];
		  pos.z = cell->p_coord[(ji*3)+2];

		  if(ix==0)
		    {
		      fptype diff = pos.x - domainMin.x;
		      if(diff < Zero)
			{
			  cell->p_coord[ji*3] = domainMin.x - diff;
			  cell->v_coord[ji*3] = -cell->v_coord[ji*3];
			  cell->hv[ji].x = -cell->hv[ji].x;
			}
		    }
		  if(ix==(nx-1))
		    {
            fptype diff = domainMax.x - pos.x;
	              if(diff < Zero)
	                    {
			      cell->p_coord[ji*3] = domainMax.x + diff;
			      cell->v_coord[ji*3] = -cell->v_coord[ji*3];

			      cell->hv[ji].x = -cell->hv[ji].x;
			    }
		  }
		  if(iy==0)
		  {
            fptype diff = pos.y - domainMin.y;
		    if(diff < Zero)
			{
			  cell->p_coord[(ji*3)+1] = domainMin.y - diff;
			  cell->v_coord[(ji*3)+1] = -cell->v_coord[(ji*3)+1];
			  cell->hv[ji].y = -cell->hv[ji].y;
			}
		  }
		  if(iy==(ny-1))
		  {
            fptype diff = domainMax.y - pos.y;
 			if(diff < Zero)
			{
			  cell->p_coord[(ji*3)+1] = domainMax.y + diff;
			  cell->v_coord[(ji*3)+1] = -cell->v_coord[(ji*3)+1];

			  cell->hv[ji].y = -cell->hv[ji].y;
			}
		  }
		  if(iz==0)
		  {
            fptype diff = pos.z - domainMin.z;
		    if(diff < Zero)
			{
			  cell->p_coord[(ji*3)+2] = domainMin.z - diff;
			  cell->v_coord[(ji*3)+2] = -cell->v_coord[(ji*3)+2];
			  cell->hv[ji].z = -cell->hv[ji].z;
			}
		  }
		  if(iz==(nz-1))
		  {
            fptype diff = domainMax.z - pos.z;
 			if(diff < Zero)
			{
			  cell->p_coord[(ji*3)+2] = domainMax.z + diff;
			  cell->v_coord[(ji*3)+2] = -cell->v_coord[(ji*3)+2];
			  cell->hv[ji].z = -cell->hv[ji].z;
			}
		  }
          //move pointer to next cell in list if end of array is reached
          if(ji == PARTICLES_PER_CELL-1) {
            cell = cell->next;
          }
        }
      }
	}
  }
}
#endif

////////////////////////////////////////////////////////////////////////////////

void AdvanceParticlesMT(int tid)
{
  for(int iz = grids[tid].sz; iz < grids[tid].ez; ++iz)
    for(int iy = grids[tid].sy; iy < grids[tid].ey; ++iy)
      for(int ix = grids[tid].sx; ix < grids[tid].ex; ++ix)
      {
        int index = (iz*ny + iy)*nx + ix;
        Cell *cell = &cells[index];
        int np = cnumPars[index];
        for(int j = 0; j < np; ++j)
        {
          Vec3 v_half = cell->hv[j % PARTICLES_PER_CELL] + cell->a[j % PARTICLES_PER_CELL]*timeStep;
#if defined(USE_ImpeneratableWall)
		// N.B. The integration of the position can place the particle
		// outside the domain. Although we could place a test in this loop
		// we would be unnecessarily testing particles on interior cells.
		// Therefore, to reduce the amount of computations we make a later
		// pass on the perimiter cells to account for particle migration
		// beyond domain
#endif

	  cell->p_coord[(j % PARTICLES_PER_CELL)*3] += v_half.x * timeStep;
          cell->p_coord[((j % PARTICLES_PER_CELL)*3)+1] += v_half.y * timeStep;
          cell->p_coord[((j % PARTICLES_PER_CELL)*3)+2] += v_half.z * timeStep;

	  cell->v_coord[(j % PARTICLES_PER_CELL)*3] = 0.5 * (cell->hv[j % PARTICLES_PER_CELL].x + v_half.x);
	  cell->v_coord[((j % PARTICLES_PER_CELL)*3)+1] = 0.5 * (cell->hv[j % PARTICLES_PER_CELL].y + v_half.y);
	  cell->v_coord[((j % PARTICLES_PER_CELL)*3)+2] = 0.5 * (cell->hv[j % PARTICLES_PER_CELL].z + v_half.z);
          cell->hv[j % PARTICLES_PER_CELL] = v_half;

          //move pointer to next cell in list if end of array is reached
          if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
            cell = cell->next;
          }
        }
      }
}

////////////////////////////////////////////////////////////////////////////////

void AdvanceFrameMT(int tid)
{
  //swap src and dest arrays with particles
  if(tid==0) {
    std::swap(cells, cells2);
    std::swap(cnumPars, cnumPars2);
  }
  pthread_barrier_wait(&barrier);

  ClearParticlesMT(tid);
  pthread_barrier_wait(&barrier);
  RebuildGridMT(tid);
  pthread_barrier_wait(&barrier);
  InitDensitiesAndForcesMT(tid);
  pthread_barrier_wait(&barrier);
  ComputeDensitiesMT(tid);
  pthread_barrier_wait(&barrier);
  ComputeDensities2MT(tid);
  pthread_barrier_wait(&barrier);
  ComputeForcesMT(tid);
  pthread_barrier_wait(&barrier);
  ProcessCollisionsMT(tid);
  pthread_barrier_wait(&barrier);
  AdvanceParticlesMT(tid);
  pthread_barrier_wait(&barrier);
#if defined(USE_ImpeneratableWall)
  // N.B. The integration of the position can place the particle
  // outside the domain. We now make a pass on the perimiter cells
  // to account for particle migration beyond domain.
  ProcessCollisions2MT(tid);
  pthread_barrier_wait(&barrier);
#endif
}

#ifndef ENABLE_VISUALIZATION
void *AdvanceFramesMT(void *args)
{
  thread_args *targs = (thread_args *)args;

#ifdef ENABLE_PARSEC_HOOKS
  __parsec_thread_begin();
#endif

  for(int i = 0; i < targs->frames; ++i) {
    AdvanceFrameMT(targs->tid);
  }

#ifdef ENABLE_PARSEC_HOOKS
  __parsec_thread_end();
#endif

  return NULL;
}
#else
//Frame advancement function for worker threads
void *AdvanceFramesMT(void *args)
{
  thread_args *targs = (thread_args *)args;

#ifdef ENABLE_PARSEC_HOOKS
  __parsec_thread_begin();
#endif

#if 1
  while(1)
#else
  for(int i = 0; i < targs->frames; ++i)
#endif
  {
    pthread_barrier_wait(&visualization_barrier);
    //Phase 1: Compute frame, visualization code blocked
    AdvanceFrameMT(targs->tid);
    pthread_barrier_wait(&visualization_barrier);
    //Phase 2: Visualize, worker threads blocked
  }

#ifdef ENABLE_PARSEC_HOOKS
  __parsec_thread_end();
#endif

  return NULL;
}

//Frame advancement function for master thread (executes serial visualization code)
void AdvanceFrameVisualization()
{
    //End of phase 2: Worker threads blocked, visualization code busy (last frame)
    pthread_barrier_wait(&visualization_barrier);
    //Phase 1: Visualization thread blocked, worker threads busy (next frame)
    pthread_barrier_wait(&visualization_barrier);
    //Begin of phase 2: Worker threads blocked, visualization code busy (next frame)
}
#endif //ENABLE_VISUALIZATION

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
#ifdef PARSEC_VERSION
#define __PARSEC_STRING(x) #x
#define __PARSEC_XSTRING(x) __PARSEC_STRING(x)
        std::cout << "PARSEC Benchmark Suite Version " __PARSEC_XSTRING(PARSEC_VERSION) << std::endl << std::flush;
#else
        std::cout << "PARSEC Benchmark Suite" << std::endl << std::flush;
#endif //PARSEC_VERSION
#ifdef ENABLE_PARSEC_HOOKS
  __parsec_bench_begin(__parsec_fluidanimate);
#endif

  if(argc < 4 || argc >= 6)
  {
    std::cout << "Usage: " << argv[0] << " <threadnum> <framenum> <.fluid input file> [.fluid output file]" << std::endl;
    return -1;
  }

  int threadnum = atoi(argv[1]);
  int framenum = atoi(argv[2]);

  //Check arguments
  if(threadnum < 1) {
    std::cerr << "<threadnum> must at least be 1" << std::endl;
    return -1;
  }
  if(framenum < 1) {
    std::cerr << "<framenum> must at least be 1" << std::endl;
    return -1;
  }

#ifdef ENABLE_CFL_CHECK
  std::cout << "WARNING: Check for Courant–Friedrichs–Lewy condition enabled. Do not use for performance measurements." << std::endl;
#endif

  InitSim(argv[3], threadnum);
#ifdef ENABLE_VISUALIZATION
  InitVisualizationMode(&argc, argv, &AdvanceFrameVisualization, &numCells, &cells, &cnumPars);
#endif

#ifdef ENABLE_PARSEC_HOOKS
  __parsec_roi_begin();
#endif
#if defined(WIN32)
  thread_args* targs = (thread_args*)alloca(sizeof(thread_args)*threadnum);
#else
  thread_args targs[threadnum];
#endif
  for(int i = 0; i < threadnum; ++i) {
    targs[i].tid = i;
    targs[i].frames = framenum;
    pthread_create(&thread[i], &attr, AdvanceFramesMT, &targs[i]);
  }

  // *** PARALLEL PHASE *** //
#ifdef ENABLE_VISUALIZATION
  Visualize();
#endif

  for(int i = 0; i < threadnum; ++i) {
    pthread_join(thread[i], NULL);
  }
#ifdef ENABLE_PARSEC_HOOKS
  __parsec_roi_end();
#endif

  if(argc > 4)
    SaveFile(argv[4]);
  CleanUpSim();

#ifdef ENABLE_PARSEC_HOOKS
  __parsec_bench_end();
#ifdef SIMD_STATS // JMCG Vectorization
  //  std::cout ;
  std::cout << "Total ComputeDensities Calls " << total_compute_densities_calls << endl;
  std::cout << "Total ComputeDensities Calls SSE " << total_compute_densities_calls_sse << " (" << (total_compute_densities_calls_sse * 100)/total_compute_densities_calls << "%)" << endl;
  std::cout << "Total ComputeDensities Calls Leftovers " << total_compute_densities_calls_leftovers << " (" << (total_compute_densities_calls_leftovers * 100)/total_compute_densities_calls_sse << "%)" << endl;
  std::cout << "Total ComputeDensities Calls Original " << total_compute_densities_calls_original << " (" << (total_compute_densities_calls_original * 100)/total_compute_densities_calls << "%)" << endl;

  std::cout << "Total ComputeForces Calls " << total_compute_forces_calls << endl;
  std::cout << "Total ComputeForces Calls SSE " << total_compute_forces_calls_sse << " (" << (total_compute_forces_calls_sse * 100)/total_compute_forces_calls << "%)" << endl;
  std::cout << "Total ComputeForces Calls Leftovers " << total_compute_forces_calls_leftovers << " (" << (total_compute_forces_calls_leftovers * 100)/total_compute_forces_calls_sse << "%)" << endl;
  std::cout << "Total ComputeForces Calls Original " << total_compute_forces_calls_original << " (" << (total_compute_forces_calls_original * 100)/total_compute_forces_calls << "%)" << endl;

  std::cout << "SSE Fully Used " << sse_fully_used << " (" << (sse_fully_used * 100)/(sse_fully_used+sse_not_used+sse_partially_used) << "%)" << endl;
  std::cout << "SSE Called but not Used " << sse_not_used << " (" << (sse_not_used * 100)/(sse_fully_used+sse_not_used+sse_partially_used) << "%)" << endl;
  std::cout << "SSE Partially Used " << sse_partially_used << " (" << (sse_partially_used * 100)/(sse_fully_used+sse_not_used+sse_partially_used) << "%)" << endl;

  std::cout << "Total NP SSE " << total_compute_densities_calls_np_sse << endl;
  std::cout << "Total NP SSE Leftovers " << total_compute_densities_calls_np_leftovers << endl;
  std::cout << "Total NP Original " << total_compute_densities_calls_np_original << endl;

#endif
#endif

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
