// annealer_thread.cpp
//
// Created by Daniel Schwartz-Narbonne on 14/04/07.
//
// Copyright 2007-2008 Princeton University
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

// SIMD Version by Juan M. Cebrian, NTNU - 2013. (modifications under JMCG tag)

#include "annealer_thread.h"
#include <cassert>
#include "location_t.h"
#include "annealer_types.h"
#include "netlist_elem.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include "rng.h"
#include <string.h>


#ifdef ENABLE_THREADS
#include <pthread.h>
#endif

using std::cout;
using std::endl;


//*****************************************************************************************
//
//*****************************************************************************************
// JMCG Clustering and SIMD included to this function
void annealer_thread::Run()
{
	int accepted_good_moves=0;
	int accepted_bad_moves=-1;
	double T = _start_temp;
	Rng rng; //store of randomness

#ifndef USE_CLUSTERING
	long a_id;
	long b_id;
	netlist_elem* a;
	netlist_elem* b = _netlist->get_random_element(&b_id, NO_MATCHING_ELEMENT, &rng);
#else

#if defined(PARSEC_USE_SSE) || (PARSEC_USE_NEON)
        __attribute__((aligned (16))) float netelement_loc_array[CLUSTER_ITEMS * (SIMD_WIDTH/2)];
#else
#ifdef PARSEC_USE_AVX
        __attribute__((aligned (32))) float netelement_loc_array[CLUSTER_ITEMS * (SIMD_WIDTH/2)];
#else
	float netelement_loc_array[1];
#endif
#endif

	// JMCG advance has to be tuned to produce similar quality results based on cluster size,
	// this size seems to work properly for clusters of 64 items
	int advance = ((CLUSTER_ITEMS/4)*(CLUSTER_ITEMS/4));

	routing_cost_t** delta_cost_cluster = new routing_cost_t*[CLUSTER_ITEMS];
	int items_to_keep = CLUSTER_ITEMS/2;
	long * cluster_ids = new long[CLUSTER_ITEMS]; //(long *)calloc(sizeof(long) * CLUSTER_ITEMS);
	netlist_elem** netlist_element_cluster = new netlist_elem*[CLUSTER_ITEMS];// (netlist_elem**)calloc(sizeof(netlist_elem*) * CLUSTER_ITEMS);
	for (int i = 0; i< CLUSTER_ITEMS ; i++) {
	  delta_cost_cluster[i] = new routing_cost_t[CLUSTER_ITEMS](); // The () is supposed to init to 0    (routing_cost_t *)calloc(sizeof(routing_cost_t) * CLUSTER_ITEMS);
    	  netlist_element_cluster[i] = _netlist->get_random_element(&(cluster_ids[i]), NO_MATCHING_ELEMENT, &rng);
#ifdef SIMD_WIDTH
	  memcpy(&(netelement_loc_array[i*2]), &(netlist_element_cluster[i]->present_loc.Get()->x_y[0]), sizeof(float) * 2);
#endif
	}

#endif
	int temp_steps_completed=0;
	while(keep_going(temp_steps_completed, accepted_good_moves, accepted_bad_moves)){
		T = T / 1.5;
		accepted_good_moves = 0;
		accepted_bad_moves = 0;

#ifndef USE_CLUSTERING
		// Original code
		for (int i = 0; i < _moves_per_thread_temp; i++){
		  //get a new element. Only get one new element, so that reuse should help the cache
		  a = b;
		  a_id = b_id;
		  b = _netlist->get_random_element(&b_id, a_id, &rng);

		  routing_cost_t delta_cost = calculate_delta_routing_cost(a,b);
		  move_decision_t is_good_move = accept_move(delta_cost, T, &rng);

		  //make the move, and update stats:
		  if (is_good_move == move_decision_accepted_bad){
		    accepted_bad_moves++;
		    _netlist->swap_locations(a,b);
		  } else if (is_good_move == move_decision_accepted_good){
		    accepted_good_moves++;
		    _netlist->swap_locations(a,b);
		  } else if (is_good_move == move_decision_rejected){
		    //no need to do anything for a rejected move
		  }
#else // USE_CLUSTERING
		  // Clustered code JMCG
		  for (int i = 0; i < _moves_per_thread_temp; i += advance){
		    // Get new random elements
#if 1
		    // keeps "items_to_keep" elements to increase data locality // JMCG
#ifdef SIMD_WIDTH
		    _netlist->update_random_elements(cluster_ids, netlist_element_cluster, &rng, items_to_keep, &netelement_loc_array[0]);
#else
		    _netlist->update_random_elements(cluster_ids, netlist_element_cluster, &rng, items_to_keep, NULL);
#endif
#else // if 1
		    // Complete random, may repeat items // JMCG
		    for (int i = 0; i< CLUSTER_ITEMS ; i++) {
		      netlist_element_cluster[i] = _netlist->get_random_element(&(cluster_ids[i]), NO_MATCHING_ELEMENT, &rng);
#ifdef SIMD_WIDTH
		      memcpy(&(netelement_loc_array[i*2]), &(netlist_element_cluster[i]->present_loc.Get()->x_y[0]), sizeof(float) * 2);
#endif
		    }
#endif // if 1
		    // calculate routing costs for all combinations
		    calculate_delta_routing_costs_cluster(netlist_element_cluster, delta_cost_cluster, &netelement_loc_array[0]);
		    // determine best movements and swap them, increment move counters
		    optimize_swapping(netlist_element_cluster, delta_cost_cluster, T, &rng, &accepted_good_moves, &accepted_bad_moves, &netelement_loc_array[0]);
#endif // USE_CLUSTERING
		  }
		temp_steps_completed++;

#ifdef ENABLE_THREADS
		pthread_barrier_wait(&_barrier);
#endif
		}
	}

//*****************************************************************************************
// JMCG Optimal Swapping of Items. Take too long for big clusters
// In addition, if the stop condition of the benchmark is a number of
// fixed iterations, the quality of the solution does not change much from the
// "fast" version.
// This functions is called recursively invalidating the columns and rows of the selected
// items to swap (setting costs to 0)
//*****************************************************************************************

int annealer_thread::get_acceptable_moves(routing_cost_t** delta_cost, int *solution) {
  int best_move_items = 0;
  int temp_move_items = 0;

  int* temporal_solution = new int[CLUSTER_ITEMS];
  int* temporal_best_solution = new int[CLUSTER_ITEMS];

  memcpy(temporal_best_solution, solution, sizeof(int) * CLUSTER_ITEMS);

  routing_cost_t** clone;

  clone = new routing_cost_t*[CLUSTER_ITEMS];// malloc(length*sizeof(char*));

  for (int i = 0; i< CLUSTER_ITEMS; i++) {
    clone[i] = new routing_cost_t[CLUSTER_ITEMS];
  }

  // Moving clone to stack instead of memory
  //  routing_cost_t clone[CLUSTER_ITEMS][CLUSTER_ITEMS];

  for(int i = 0; i < CLUSTER_ITEMS; i++) {
    for(int j = i+1; j < CLUSTER_ITEMS; j++) {
      if(delta_cost[i][j] < 0) {
#ifdef DEBUG_SIMD
	cout << "Delta Cost " << delta_cost[i][j] << " < 0" << endl;
#endif
	// Clone original delta cost matrix
	for ( int k = 0; k < CLUSTER_ITEMS; k++ ){
	  memcpy(clone[k], delta_cost[k], sizeof(routing_cost_t) * CLUSTER_ITEMS);
	}
#ifdef DEBUG_SIMD
	cout << "Clone created" << endl;
#endif
	// Fill [i][*], [*][j], [j][*] and [*][i] with 0, as they are invalidated by this selected item

	std::fill(clone[i], clone[i]+CLUSTER_ITEMS, 0);
	std::fill(clone[j], clone[j]+CLUSTER_ITEMS, 0);

	for (int jj = 0; jj < CLUSTER_ITEMS; jj++) {
	  clone[jj][i] = 0;
	  clone[jj][j] = 0;
	}

#ifdef DEBUG_SIMD
	cout << "Clone Filled with 0" << endl;
	cout << "Reduced Clone Cost Matrix :" << endl;
	for (int o = 0;o<CLUSTER_ITEMS;o++) {
	  for (int p = 0; p<CLUSTER_ITEMS;p++) {
	    cout << clone[o][p] << " ";
	  }
	  cout << endl;
	}
#endif
	// recursive call with new input
	memcpy(temporal_solution, solution, sizeof(int) * CLUSTER_ITEMS);
	temp_move_items = 1 + get_acceptable_moves(&clone[0], temporal_solution);
	// Update best solution
	if (temp_move_items > best_move_items) {
	  best_move_items = temp_move_items;
	  memcpy(temporal_best_solution, temporal_solution, sizeof(int) * CLUSTER_ITEMS);
	  temporal_best_solution[i] = j;
	  temporal_best_solution[j] = i;
	}
      }
    }
  }

  memcpy(solution, temporal_best_solution, sizeof(int) * CLUSTER_ITEMS);

  delete [] temporal_solution;

  // Free clone

  for ( int k = 0; k < CLUSTER_ITEMS; k++ ){
    delete [] clone[k];
  }
  delete [] clone;

  return best_move_items;
}


/***********************************************************************/
// JMCG Suboptimal selection of swapping points, not usefull
// when the stop condition depends on the quality of the solution
/***********************************************************************/
int annealer_thread::get_acceptable_moves_fast(routing_cost_t** delta_cost, int *solution) {

  int moves = 0;

  for(int i = 0; i < CLUSTER_ITEMS; i++) {
    for(int j = i+1; j < CLUSTER_ITEMS; j++) {
      if(delta_cost[i][j] < 0) {
#ifdef DEBUG_SIMD
	cout << "Delta Cost " << delta_cost[i][j] << " < 0" << endl;
#endif
	// Fill [i][*], [*][j], [j][*] and [*][i] with 0, as they are invalidated by this selected item
	std::fill(delta_cost[i], delta_cost[i]+CLUSTER_ITEMS, 0);
	std::fill(delta_cost[j], delta_cost[j]+CLUSTER_ITEMS, 0);
	for (int jj = 0; jj < CLUSTER_ITEMS; jj++) {
	  delta_cost[jj][i] = 0;
	  delta_cost[jj][j] = 0;
	}

#ifdef DEBUG_SIMD
	cout << "Clone Filled with 0" << endl;
	cout << "Reduced Clone Cost Matrix :" << endl;
	for (int o = 0;o<CLUSTER_ITEMS;o++) {
	  for (int p = 0; p<CLUSTER_ITEMS;p++) {
	    cout << delta_cost[o][p] << " ";
	  }
	  cout << endl;
	}
#endif

	// Update best solution
	solution[i] = j;
	solution[j] = i;
	moves++;

	}
      }
    }

  return moves;
}


/************************************************************************************/

 void annealer_thread::optimize_swapping(netlist_elem** input, routing_cost_t** delta_cost, double T, Rng* rng, int* good_moves, int* bad_moves, float* netelement_loc_array)
{
  int* solution = new int[CLUSTER_ITEMS];
  int j;
  float x, y;

  for (int i = 0; i< CLUSTER_ITEMS; i++){
    solution[i] = i;
  }

  /* JMCG BEGIN */
  if (_number_temp_steps == -1) {
    // If our stop condition is a threshold use optimal solution search
    *good_moves += get_acceptable_moves(delta_cost, solution);
  } else {
    // if not (brute force), just select first change available
    *good_moves += get_acceptable_moves_fast(delta_cost, solution);
  }

  for (int i = 0; i < CLUSTER_ITEMS; i++) {
    if (solution[i] == i) {
      //cout << "Bad move chaging i " << i;
      for(j = i+1; j < CLUSTER_ITEMS; j++ ) {
	if (solution[j] == j) {
	  // cout << " For j " << j << endl;
	  double random_value = rng->drand();
	  double boltzman = exp(- delta_cost[i][j]/T);
	  if (boltzman > random_value){
	    *bad_moves += 1;
	    solution[i] = j;
	    solution[j] = i;
	  }
	  i = j+1;
	  break;
	}
      }
    }
  }

#ifdef DEBUG_SIMD
  cout << "End badmoves " << endl;

  cout << "Good moves " << *good_moves << " Bad moves: " << *bad_moves << " Solution: " << endl;
  for (int i = 0; i< CLUSTER_ITEMS ; i++) {
    cout << solution[i] << " ";
  }
  cout << endl;
#endif

  for (int i = 0; i< CLUSTER_ITEMS; i++ ){
    if (solution[i] != i) {
      if (input[i] != input[solution[i]]) {
	_netlist->swap_locations(input[i],input[solution[i]]);

#ifdef SIMD_WIDTH
	x = netelement_loc_array[i*2];
	netelement_loc_array[i*2] = netelement_loc_array[solution[i]*2];
	netelement_loc_array[solution[i]*2] = x;

	y = netelement_loc_array[(i*2)+1];
	netelement_loc_array[(i*2)+1] = netelement_loc_array[(solution[i]*2)+1];
	netelement_loc_array[(solution[i]*2)+1] = y;
#endif

	solution[solution[i]] = solution[i];
	solution[i] = i;

      }
    }
  }

  delete [] solution;

  return;
}
  /* JMCG END */

//*****************************************************************************************
//
//*****************************************************************************************
annealer_thread::move_decision_t annealer_thread::accept_move(routing_cost_t delta_cost, double T, Rng* rng)
{
	//always accept moves that lower the cost function
	if (delta_cost < 0){
		return move_decision_accepted_good;
	} else {
		double random_value = rng->drand();
		double boltzman = exp(- delta_cost/T);
		if (boltzman > random_value){
			return move_decision_accepted_bad;
		} else {
			return move_decision_rejected;
		}
	}
}


//*****************************************************************************************
//  If get turns out to be expensive, I can reduce the # by passing it into the swap cost fcn
//*****************************************************************************************
routing_cost_t annealer_thread::calculate_delta_routing_cost(netlist_elem* a, netlist_elem* b)
{
	location_t* a_loc = a->present_loc.Get();
	location_t* b_loc = b->present_loc.Get();

#ifdef PARSEC_USE_SSE
	// Non-clustered SSE
	// This is only implemented for SSE, since it did not provide much speedup
        __attribute__((aligned (16))) float loc_pair[4];

        loc_pair[0] = (float)b_loc->x;
        loc_pair[1] = (float)b_loc->y;
        loc_pair[2] = (float)a_loc->x;
        loc_pair[3] = (float)a_loc->y;

        routing_cost_t delta_cost = a->swap_cost_sse(&(loc_pair[0]));

        loc_pair[0] = (float)a_loc->x;
        loc_pair[1] = (float)a_loc->y;
        loc_pair[2] = (float)b_loc->x;
        loc_pair[3] = (float)b_loc->y;

        delta_cost += b->swap_cost_sse(&(loc_pair[0]));
#else
        routing_cost_t delta_cost = a->swap_cost(a_loc, b_loc);
        delta_cost += b->swap_cost(b_loc, a_loc);
#endif

	return delta_cost;
}

/**********************************************************************************************/
//
/**********************************************************************************************/

 void annealer_thread::calculate_delta_routing_costs_cluster(netlist_elem** input, routing_cost_t **delta_cost_matrix_complete, float* netelement_loc_array)
{

  location_t* cluster_loc[CLUSTER_ITEMS];
  routing_cost_t *cost_matrix[CLUSTER_ITEMS];

  int delta_index = 0;


  for(int i = 0; i < CLUSTER_ITEMS; i++) {
    cost_matrix[i] = new routing_cost_t[CLUSTER_ITEMS](); // The () is supposed to init to 0    (routing_cost_t *)calloc(sizeof(routing_cost_t) * CLUSTER_ITEMS);
    cluster_loc[i] = (input[i])->present_loc.Get();
  }

#ifdef SIMD_WIDTH
  // JMCG Obtain cost matrix, SIMD
  for(int i = 0; i < CLUSTER_ITEMS; i++) {
    (input[i])->swap_cost_block_simd(&cluster_loc[0], cost_matrix[i], &netelement_loc_array[0]);
  }

#else
  // JMCG Obtain cost matrix, NO SIMD
  for(int i = 0; i < CLUSTER_ITEMS; i++) {
    (input[i])->swap_cost_block(&cluster_loc[0], cost_matrix[i]);
  }

#endif

  // Calculate delta costs
  // We can try to vectorize this at some point
  for(int i = 0; i < CLUSTER_ITEMS; i++) {
    for(int j = (i+1); j<CLUSTER_ITEMS; j++) {
      //cout << "Operation " << cost_matrix[i][j] << " - " << cost_matrix[i][i] << " + " << cost_matrix[j][i] << " - " << cost_matrix[j][j] << endl;
      delta_cost_matrix_complete[i][j] = (cost_matrix[i][j] - cost_matrix[i][i]) + (cost_matrix[j][i] - cost_matrix[j][j]); // JMCG this produces the same yes_swap and no_swap as the original swap_cost code
#if DEBUG_SIMD
      delta_index++;
#endif
    }
  }

#if DEBUG_SIMD
  cout << "Max Delta Index: " << delta_index << endl;
  cout << "Cost Matrix :" << endl;
  for (int i = 0;i<CLUSTER_ITEMS;i++) {
    for (int j = 0; j<CLUSTER_ITEMS;j++) {
      cout << cost_matrix[i][j] << " ";
    }
    cout << endl;
  }
  cout << "Delta Cost Matrix :" << endl;
  for (int i = 0;i<CLUSTER_ITEMS;i++) {
    for (int j = 0; j<CLUSTER_ITEMS;j++) {
      cout << delta_cost_matrix_complete[i][j] << " ";
    }
    cout << endl;
  }
#endif

  for(int i = 0; i < CLUSTER_ITEMS; i++) {
    delete [] cost_matrix[i];
  }

  return;
}


//*****************************************************************************************
//  Check whether design has converged or maximum number of steps has reached
//*****************************************************************************************
bool annealer_thread::keep_going(int temp_steps_completed, int accepted_good_moves, int accepted_bad_moves)
{
	bool rv;

	if(_number_temp_steps == -1) {
		//run until design converges
		rv = _keep_going_global_flag && (accepted_good_moves > accepted_bad_moves);
		if(!rv) _keep_going_global_flag = false; // signal we have converged
	} else {
		//run a fixed amount of steps
		rv = temp_steps_completed < _number_temp_steps;
	}

	return rv;
}

