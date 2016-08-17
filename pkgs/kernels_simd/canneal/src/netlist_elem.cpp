// netlist_elem.cpp
//
// Created by Daniel Schwartz-Narbonne on 14/04/07.
//
// Copyright 2007 Princeton University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.


/* JMCG BEGIN */

#include "simd_header.h"
#include "clustering.h"

// #define DEBUG_SIMD

/* JMCG END */

#include <assert.h>
#include <math.h>

#include "annealer_types.h"
#include "location_t.h"
#include "netlist_elem.h"

#include <iostream>

using namespace std;




netlist_elem::netlist_elem()
:present_loc(NULL)//start with the present_loc as nothing at all.  Filled in later by the netlist
{
}

//*****************************************************************************************
// Calculates the routing cost using the manhatten distance
// I make sure to get the pointer in one operation, and then use it
// SYNC: Do i need to make this an atomic operation?  i.e. are there misaligned memoery issues that can cause this to fail
//       even if I have atomic writes?
//*****************************************************************************************
routing_cost_t netlist_elem::routing_cost_given_loc(location_t loc)
{
	routing_cost_t fanin_cost = 0;
	routing_cost_t fanout_cost = 0;

	for (int i = 0; i< fanin.size(); ++i){
		location_t* fanin_loc = fanin[i]->present_loc.Get();
		fanin_cost += fabs(loc.x - fanin_loc->x);
		fanin_cost += fabs(loc.y - fanin_loc->y);
	}

	for (int i = 0; i< fanout.size(); ++i){
		location_t* fanout_loc = fanout[i]->present_loc.Get();
		fanout_cost += fabs(loc.x - fanout_loc->x);
		fanout_cost += fabs(loc.y - fanout_loc->y);
	}

	routing_cost_t total_cost = fanin_cost + fanout_cost;
	return total_cost;
}

//*****************************************************************************************
//  Get the cost change of swapping from our present location to a new location
//*****************************************************************************************
routing_cost_t netlist_elem::swap_cost(location_t* old_loc, location_t* new_loc)
{
	routing_cost_t no_swap = 0;
	routing_cost_t yes_swap = 0;

	for (int i = 0; i< fanin.size(); ++i){
		location_t* fanin_loc = fanin[i]->present_loc.Get();
		no_swap += fabs(old_loc->x - fanin_loc->x);
		no_swap += fabs(old_loc->y - fanin_loc->y);

		yes_swap += fabs(new_loc->x - fanin_loc->x);
		yes_swap += fabs(new_loc->y - fanin_loc->y);
	}

	for (int i = 0; i< fanout.size(); ++i){
		location_t* fanout_loc = fanout[i]->present_loc.Get();
		no_swap += fabs(old_loc->x - fanout_loc->x);
		no_swap += fabs(old_loc->y - fanout_loc->y);

		yes_swap += fabs(new_loc->x - fanout_loc->x);
		yes_swap += fabs(new_loc->y - fanout_loc->y);
	}

	return yes_swap - no_swap;
}

/* JMCG BEGIN */

// Simple swap cost sse implementation, no speedup due to high cache misses and
// small computations performed over the loaded data

#ifdef PARSEC_USE_SSE
routing_cost_t netlist_elem::swap_cost_sse(float * loc_pair)
{

  /* routing_cost_t is double */

  /* Low will be no_swap and high will be yes_swap*/

  //  __m128d swapcost = _mm_setzero_pd();
  __m128 swapcost = _mm_setzero_ps(); // JMCG Changing from double to int, location differences are really small, should not overflow
  __m128 present_loc;
  __m128 composite_loc = _mm_load_ps(loc_pair);


  // Take this out of stack, it looks like the stack cannot aligned
  __attribute__((aligned (16))) float result_array[4];

  for (int i = 0; i< fanin.size(); ++i){
    location_t* fanin_loc = fanin[i]->present_loc.Get();

    /* presentloc has x y x y */
    present_loc = _mm_load_ps(&(fanin_loc->x_y[0]));

    /* composite_loc contains new_loc.x new_loc.y old_loc.x old_loc.y*/
    swapcost = _mm_add_ps(swapcost,_mm_abs_ps(_mm_sub_ps(composite_loc, present_loc)));

  }

  for (int i = 0; i< fanout.size(); ++i){
    location_t* fanout_loc = fanout[i]->present_loc.Get();
    present_loc = _mm_load_ps(&(fanout_loc->x_y[0]));

    swapcost = _mm_add_ps(swapcost,_mm_abs_ps(_mm_sub_ps(composite_loc, present_loc)));

  }

  /* a[0] + a[1] || a[2] + a[3] || a[0] + a[1] || a[2] + a[3] --> yes_swap | no_swap | yes_swap | no_swap
     yes_swap - no_swap | yes_swap - no_swap | yes_swap - no_swap | yes_swap - no_swap */

  swapcost = _mm_hadd_ps(swapcost, swapcost);
  swapcost = _mm_hsub_ps(swapcost, swapcost);

  _mm_store_ps(&(result_array[0]), swapcost);

  return ((routing_cost_t)result_array[0]);
}
#endif

//*****************************************************************************************
//  clustered version of the swap cost, this calculates a distance matrix, the actual swap
//  costs are computed by the caller
//*****************************************************************************************

void netlist_elem::swap_cost_block(location_t** location_cluster, routing_cost_t* cluster_costs)
{
  // Cluster costs allocated by caller and initiallized to 0

  for (int i = 0; i< fanin.size(); ++i){
    location_t* fanin_loc = fanin[i]->present_loc.Get();

    for (int j = 0; j < CLUSTER_ITEMS; j++) {
      cluster_costs[j] += fabs(location_cluster[j]->x - fanin_loc->x);
      cluster_costs[j] += fabs(location_cluster[j]->y - fanin_loc->y);
    }
  }

  for (int i = 0; i< fanout.size(); ++i){
    location_t* fanout_loc = fanout[i]->present_loc.Get();
    for (int j = 0; j < CLUSTER_ITEMS ; j++) {
      cluster_costs[j] += fabs(location_cluster[j]->x - fanout_loc->x);
      cluster_costs[j] += fabs(location_cluster[j]->y - fanout_loc->y);
    }
  }

  return;
}


//*****************************************************************************************
//  SIMD clustered version of the swap cost, this calculates a distance matrix, the actual swap
//  costs are computed by the caller
//*****************************************************************************************

#ifdef SIMD_WIDTH

void netlist_elem::swap_cost_block_simd(location_t** location_cluster, routing_cost_t* cluster_costs, float* netelement_loc_array) {
  // Cluster costs allocated by caller and initiallized to 0

  _MM_TYPE present_loc, temp;

  _MM_TYPE composite_loc[CLUSTER_ITEMS/(SIMD_WIDTH/2)];
  _MM_TYPE swapcost[CLUSTER_ITEMS/(SIMD_WIDTH/2)];

  for (int j = 0; j < CLUSTER_ITEMS/(SIMD_WIDTH/2); j++) {
    composite_loc[j] = _MM_LOADU(&(netelement_loc_array[j*SIMD_WIDTH]));
    swapcost[j] = _MM_SETZERO();
  }

  for (int i = 0; i< fanin.size(); ++i){
    location_t* fanin_loc = fanin[i]->present_loc.Get();
#if (SIMD_WIDTH > 4)
    // AVX only at this point
    present_loc = _MM_BROADCAST_128((__m128i *)(&fanin_loc->x_y[0]));
    // Alternative
    // __m128 temp_load = _mm_load_ps(&(fanin_loc->x_y[0]));
    // present_loc = _mm256_insertf128_ps(_mm256_insertf128_ps(present_loc,temp_load,0),temp_load,1);
#else
    // SSE/NEON
    present_loc = _MM_LOAD(&(fanin_loc->x_y[0]));
#endif
    // JMCG Unroll CLUSTER_ITEMS ?
#pragma unroll (CLUSTER_ITEMS/(SIMD_WIDTH/2))
    for (int j = 0; j < (CLUSTER_ITEMS/(SIMD_WIDTH/2)); j++) { // Two coordinates x,y
      swapcost[j] = _MM_ADD(swapcost[j],_MM_ABS(_MM_SUB(composite_loc[j], present_loc)));
    }
  }

  for (int i = 0; i< fanout.size(); ++i){
    location_t* fanout_loc = fanout[i]->present_loc.Get();

#if (SIMD_WIDTH > 4)
    // AVX only at this point
    present_loc = _MM_BROADCAST_128((__m128i *)(&fanout_loc->x_y[0]));
    // Alternative
    //    __m128 temp_load = _mm_load_ps(&(fanout_loc->x_y[0]));
    //    present_loc = _mm256_insertf128_ps(_mm256_insertf128_ps(present_loc,temp_load,0),temp_load,1);
#else
    // SSE/NEON
    present_loc = _MM_LOAD(&(fanout_loc->x_y[0]));
#endif

    for (int j = 0; j < (CLUSTER_ITEMS/(SIMD_WIDTH/2)); j++) {
      swapcost[j] = _MM_ADD(swapcost[j],_MM_ABS(_MM_SUB(composite_loc[j], present_loc)));
    }
  }

  // JMCG rebuild cluster_costs
  for (int j = 0, i=0; j < (CLUSTER_ITEMS/(SIMD_WIDTH/2)); j+=2) {
    temp = _MM_RHADD(swapcost[j], swapcost[j+1]);
    _MM_STOREU(&(cluster_costs[i*SIMD_WIDTH]), temp);
    i++;
  }

  return;
}

#endif


/* JMCG END */
