// Juan M. Cebrian, NTNU - 2018.

// TODO: Makefile
// g++ -O3 ../../obj/amd64-linux.gcc-hooks/cellpool.o ../../obj/amd64-linux.gcc-hooks/parsec_barrier.o combine_inputs.cpp -pthread -D_GNU_SOURCE -D__XOPEN_SOURCE=600

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
#include <map>

#include "../fluid.hpp"
#include "../cellpool.hpp"
#include "../parsec_barrier.hpp"

#include <iomanip>

#ifdef ENABLE_VISUALIZATION
#include "../fluidview.hpp"
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

int cell_divisor = 1;
int merge_files = 0;

int nx, ny, nz;    // number of grid cells in each dimension
Vec3 delta;        // cell dimensions
int numParticles = 0;
int numCells = 0;
Cell *cells = 0;
Cell *cells2 = 0;
int *cnumPars = 0;
int *cnumPars2 = 0;

Cell **last_cells = NULL; //helper array with pointers to last cell structure of "cells" array lists

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

/* Second file */

cellpool *pools_s; //each thread has its private cell pool

fptype restParticlesPerMeter_s, h_s, hSq_s;
fptype densityCoeff_s, pressureCoeff_s, viscosityCoeff_s;

int nx_s, ny_s, nz_s;    // number of grid cells in each dimension
Vec3 delta_s;        // cell dimensions
int numParticles_s = 0;
int numCells_s = 0;
Cell *cells_s = 0;
Cell *cells2_s = 0;
int *cnumPars_s = 0;
int *cnumPars2_s = 0;

Cell **last_cells_s = NULL; //helper array with pointers to last cell structure of "cells" array lists

int XDIVS_s = 1;  // number of partitions in X
int ZDIVS_s = 1;  // number of partitions in Z

#define NUM_GRIDS_s  ((XDIVS_s) * (ZDIVS_s))
#define MUTEXES_PER_CELL_s 128
#define CELL_MUTEX_ID_s 0

struct Grid *grids_s;
bool  *border_s;
pthread_attr_t attr_s;
pthread_t *thread_s;
pthread_mutex_t **mutex_s;  // used to lock cells in RebuildGrid and also particles in other functions
pthread_barrier_t barrier_s;  // global barrier used by all threads

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

void InitSim(char const *fileName, char const *fileName_s, unsigned int threadnum)
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
  pools_s = new cellpool[NUM_GRIDS];

  //Load input particles
  std::cout << "Loading file \"" << fileName << "\"..." << std::endl;
  std::ifstream file(fileName, std::ios::binary);
  if(!file) {
    std::cerr << "Error opening file. Aborting." << std::endl;
    exit(1);
  }
  std::cout << "Loading file \"" << fileName_s << "\"..." << std::endl;
  std::ifstream file_s(fileName_s, std::ios::binary);
  if(!file_s) {
    std::cerr << "Error opening file_s. Aborting." << std::endl;
    exit(1);
  }

  /*
  file.seekg(0, std::ios_base::end);
  file_s.seekg(0, std::ios_base::end);
  std::cout << " ifstream  file1 size: " << file.tellg() << std::endl;
  std::cout << " ifstream  file2 size: " << file_s.tellg() << std::endl;
  */

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

  //Always use single precision float variables b/c file format uses single precision
  float restParticlesPerMeter_le_s;
  int numParticles_le_s;
  file_s.read((char *)&restParticlesPerMeter_le_s, FILE_SIZE_FLOAT);
  file_s.read((char *)&numParticles_le_s, FILE_SIZE_INT);
  if(!isLittleEndian()) {
    restParticlesPerMeter_s = bswap_float(restParticlesPerMeter_le_s);
    numParticles_s          = bswap_int32(numParticles_le_s);
  } else {
    restParticlesPerMeter_s = restParticlesPerMeter_le_s;
    numParticles_s          = numParticles_le_s;
  }
  for(int i=0; i<NUM_GRIDS; i++) cellpool_init(&pools_s[i], numParticles_s/NUM_GRIDS);

  h_s = kernelRadiusMultiplier / restParticlesPerMeter_s;
  hSq_s = h_s*h_s;

#ifndef ENABLE_DOUBLE_PRECISION
  fptype coeff1 = 315.0 / (64.0*pi*powf(h,9.0));
  fptype coeff2 = 15.0 / (pi*powf(h,6.0));
  fptype coeff3 = 45.0 / (pi*powf(h,6.0));
  fptype coeff1_s = 315.0 / (64.0*pi*powf(h_s,9.0));
  fptype coeff2_s = 15.0 / (pi*powf(h_s,6.0));
  fptype coeff3_s = 45.0 / (pi*powf(h_s,6.0));
#else
  fptype coeff1 = 315.0 / (64.0*pi*pow(h,9.0));
  fptype coeff2 = 15.0 / (pi*pow(h,6.0));
  fptype coeff3 = 45.0 / (pi*pow(h,6.0));
  fptype coeff1_s = 315.0 / (64.0*pi*pow(h_s,9.0));
  fptype coeff2_s = 15.0 / (pi*pow(h_s,6.0));
  fptype coeff3_s = 45.0 / (pi*pow(h_s,6.0));
#endif //ENABLE_DOUBLE_PRECISION

  fptype particleMass = 0.5*doubleRestDensity / (restParticlesPerMeter*restParticlesPerMeter*restParticlesPerMeter);
  densityCoeff = particleMass * coeff1;
  pressureCoeff = 3.0*coeff2 * 0.50*stiffnessPressure * particleMass;
  viscosityCoeff = viscosity * coeff3 * particleMass;

  fptype particleMass_s = 0.5*doubleRestDensity / (restParticlesPerMeter_s*restParticlesPerMeter_s*restParticlesPerMeter_s);
  densityCoeff_s = particleMass_s * coeff1_s;
  pressureCoeff_s = 3.0*coeff2_s * 0.50*stiffnessPressure * particleMass_s;
  viscosityCoeff_s = viscosity * coeff3_s * particleMass_s;


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
  int rv5 = posix_memalign((void **)(&cells_s), CACHELINE_SIZE, sizeof(struct Cell) * numCells);
  int rv6 = posix_memalign((void **)(&cells2_s), CACHELINE_SIZE, sizeof(struct Cell) * numCells);
  int rv7 = posix_memalign((void **)(&cnumPars_s), CACHELINE_SIZE, sizeof(int) * numCells);
  int rv8 = posix_memalign((void **)(&cnumPars2_s), CACHELINE_SIZE, sizeof(int) * numCells);
  int rv9 = posix_memalign((void **)(&last_cells_s), CACHELINE_SIZE, sizeof(struct Cell *) * numCells);
  assert((rv0==0) && (rv1==0) && (rv2==0) && (rv3==0) && (rv4==0) && (rv5==0) && (rv6==0) && (rv7==0) && (rv8==0) && (rv9==0));
#endif

  // because cells and cells2 are not allocated via new
  // we construct them here
  for(int i=0; i<numCells; ++i)
  {
	  new (&cells[i]) Cell;
	  new (&cells2[i]) Cell;
	  new (&cells_s[i]) Cell;
	  new (&cells2_s[i]) Cell;
  }

  memset(cnumPars, 0, numCells*sizeof(int));
  memset(cnumPars_s, 0, numCells*sizeof(int));

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
//      std::cout << "Adding another structure first file" << std::endl;
      //     fflush(stdout);
      // getchar();
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

//  file.close();

  //Always use single precision float variables b/c file format uses single precision float
  int pool_id_s = 0;
  for(int i = 0; i < numParticles_s; ++i)
  {
    file_s.read((char *)&px, FILE_SIZE_FLOAT);
    file_s.read((char *)&py, FILE_SIZE_FLOAT);
    file_s.read((char *)&pz, FILE_SIZE_FLOAT);
    file_s.read((char *)&hvx, FILE_SIZE_FLOAT);
    file_s.read((char *)&hvy, FILE_SIZE_FLOAT);
    file_s.read((char *)&hvz, FILE_SIZE_FLOAT);
    file_s.read((char *)&vx, FILE_SIZE_FLOAT);
    file_s.read((char *)&vy, FILE_SIZE_FLOAT);
    file_s.read((char *)&vz, FILE_SIZE_FLOAT);
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
    Cell *cell_s = &cells_s[index];

    //go to last cell structure in list
    int np = cnumPars_s[index];
    while(np > PARTICLES_PER_CELL) {
      cell_s = cell_s->next;
      np = np - PARTICLES_PER_CELL;
    }
    //add another cell structure if everything full
    if( (np % PARTICLES_PER_CELL == 0) && (cnumPars_s[index] != 0) ) {
/*      std::cout << "Adding another structure second file" << std::endl;
      fflush(stdout);
      getchar();
*/
      //Get cells from pools in round-robin fashion to balance load during parallel phase
      cell_s->next = cellpool_getcell(&pools_s[pool_id]);
      pool_id_s = (pool_id_s+1) % NUM_GRIDS;
      cell_s = cell_s->next;
      np = np - PARTICLES_PER_CELL;
    }

    cell_s->p_coord[np*3] = px;
    cell_s->p_coord[(np*3)+1] = py;
    cell_s->p_coord[(np*3)+2] = pz;

    cell_s->v_coord[np*3] = vx;
    cell_s->v_coord[(np*3)+1] = vy;
    cell_s->v_coord[(np*3)+2] = vz;

    cell_s->hv[np].x = hvx;
    cell_s->hv[np].y = hvy;
    cell_s->hv[np].z = hvz;

    ++cnumPars_s[index];
  }

  map<int, int> histArray;
  std::cout << "Number of particles: " << numParticles << std::endl;
  for (int i = 0; i < numCells; i++) {
      if ( histArray.find(cnumPars[i]) == histArray.end() ) {
	histArray[cnumPars[i]] = 1;
      } else {
	histArray[cnumPars[i]]++;
      }
  }
  std::cout << "Histogram of particles per cell: ";
  for(map<int,int>::iterator it = histArray.begin(); it != histArray.end(); ++it) {
    cout << it->first << ":" << it->second << " ";
  }
  std::cout << std::endl;

  map<int, int> histArray_s;
  std::cout << "Number of particles: " << numParticles_s << std::endl;
  for (int i = 0; i < numCells; i++) {
      if ( histArray_s.find(cnumPars_s[i]) == histArray_s.end() ) {
	histArray_s[cnumPars_s[i]] = 1;
      } else {
	histArray_s[cnumPars_s[i]]++;
      }
  }
  std::cout << "Histogram of particles per cell: ";
  for(map<int,int>::iterator it = histArray_s.begin(); it != histArray_s.end(); ++it) {
    cout << it->first << ":" << it->second << " ";
  }
  std::cout << std::endl;

}

////////////////////////////////////////////////////////////////////////////////

void SaveFile(char const *fileName)
{
  std::cout << "Saving file \"" << fileName << "\"..." << std::endl;

  std::ofstream file(fileName, std::ios::binary);
  assert(file);
  restParticlesPerMeter /= cell_divisor;
  if(merge_files == 0)
    numParticles += numParticles_s;
  if(merge_files == 2)
    numParticles = numParticles_s;

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
  int count_s = 0;



  for(int i = 0; i < numCells; ++i) {
    Cell *cell = &cells[i];
    int np = cnumPars[i];
    if((merge_files == 0) || (merge_files == 1)) {
      for(int j = 0; j < np; ++j) {
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

    Cell *cell_s = &cells_s[i];
    np = cnumPars_s[i];
    if((merge_files == 0) || (merge_files == 2)) {
      for(int j = 0; j < np; ++j) {
	//Always use single precision float variables b/c file format uses single precision
	float px, py, pz, hvx, hvy, hvz, vx,vy, vz;
	if(!isLittleEndian()) {
	  px  = bswap_float((float)(cell_s->p_coord[(j % PARTICLES_PER_CELL)*3]));
	  py  = bswap_float((float)(cell_s->p_coord[((j % PARTICLES_PER_CELL)*3)+1]));
	  pz  = bswap_float((float)(cell_s->p_coord[((j % PARTICLES_PER_CELL)*3)+2]));

	  vx  = bswap_float((float)(cell_s->v_coord[(j % PARTICLES_PER_CELL)*3]));
	  vy  = bswap_float((float)(cell_s->v_coord[((j % PARTICLES_PER_CELL)*3)+1]));
	  vz  = bswap_float((float)(cell_s->v_coord[((j % PARTICLES_PER_CELL)*3)+2]));

	  hvx = bswap_float((float)(cell_s->hv[j % PARTICLES_PER_CELL].x));
	  hvy = bswap_float((float)(cell_s->hv[j % PARTICLES_PER_CELL].y));
	  hvz = bswap_float((float)(cell_s->hv[j % PARTICLES_PER_CELL].z));
	} else {
	  px  = (float)(cell_s->p_coord[(j % PARTICLES_PER_CELL)*3]);
	  py  = (float)(cell_s->p_coord[((j % PARTICLES_PER_CELL)*3)+1]);
	  pz  = (float)(cell_s->p_coord[((j % PARTICLES_PER_CELL)*3)+2]);

	  vx  = (float)(cell_s->v_coord[(j % PARTICLES_PER_CELL)*3]);
	  vy  = (float)(cell_s->v_coord[((j % PARTICLES_PER_CELL)*3)+1]);
	  vz  = (float)(cell_s->v_coord[((j % PARTICLES_PER_CELL)*3)+2]);

	  hvx = (float)(cell_s->hv[j % PARTICLES_PER_CELL].x);
	  hvy = (float)(cell_s->hv[j % PARTICLES_PER_CELL].y);
	  hvz = (float)(cell_s->hv[j % PARTICLES_PER_CELL].z);
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
	++count_s;

	//move pointer to next cell in list if end of array is reached
	if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
	  cell_s = cell_s->next;
	}
      }
    }

  }
  std::cout << "Saving " << count+count_s << " should be: " << numParticles << std::endl;
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

int main(int argc, char *argv[])
{
  std::cout << argc << std::endl;
  if(argc < 6 || argc >= 7)
  {
    std::cout << "Usage: " << argv[0] << " <.fluid input file> <.fluid input file2> [.fluid output file] [number of particles of output file (0 = addition of inputs, 1 = particle count equal input1, 2 = particle count equals input2] [cell divisor (any number >=1)]" << std::endl;
    std::cout << "Note: We recomend to run fluidanimate with default inputs and use the outputs as inputs for the combination, since that would have moved the particles to new positions" << std::endl;
    std::cout << "Example: " << argv[0] << "in_500K.fluid out_500K.fluid merge.fluid 0 2 # Will create a combine file with 1M particles, but half the number of cells (thus more density)" << std::endl;
    std::cout << "Example2: " << argv[0] << "in_500K.fluid in_500K.fluid merge.fluid 1 2 # Will create a file with 500K particles, but half the number of cells (thus more density)" << std::endl;
    return -1;
  }

  cell_divisor = atoi(argv[5]);
  merge_files = atoi(argv[4]);

  InitSim(argv[1], argv[2], 1);

  if(argc > 2)
    SaveFile(argv[3]);

  CleanUpSim();
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
