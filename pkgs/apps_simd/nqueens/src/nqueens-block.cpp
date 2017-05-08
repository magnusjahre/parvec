/**********************************************************/
/* This code is for PLDI-15 Artifact Evaluation only      */ 
/* and will be released with further copyright information*/ 
/* File: SSE block w reexpansion of nqueens               */
/**********************************************************/

/********************************************
    
    Modified nqueens code that supports 
    both reexp and no-reexp configurations 
    running under scalar, SSEand AVX

*********************************************/

#include <iostream>
#include <fstream>
#include "harness.h"

/* CDF START */
#if defined (PARSEC_USE_SSE) || defined (PARSEC_USE_AVX)
//#define SCALAR 0
#include "simd.h"	// must precede block.h
#include "block-sse.h"
#else
#define SCALAR 1
#include "block.h"
#endif

#ifdef ENABLE_PARSEC_HOOKS
#include "simd_defines.h"
#include <hooks.h>
#endif
/* CDF END */

/*#ifdef BLOCK_PROFILE
#include "blockprofiler.h"
BlockProfiler profiler;
#endif*/

#ifdef SCALAR
#ifdef TRACK_TRAVERSALS
	uint64_t work = 0;	
#endif
#endif

#ifdef REEXP
#ifdef SCALAR 
int* expand_condition_table_10;
int* expand_condition_table_11;
int* expand_condition_table_12;
int* expand_condition_table_13;
int* expand_condition_table_14;
#endif
#ifndef SCALAR
int* expand_condition_table;
#endif
int expand_condition;
#endif

using namespace std;

//int _expandDepth = 0;
int _expandSize = D_MAX_BLOCK_SIZE;

#ifdef REEXP
int dynamic_reexpand_count = 0;
int g_is_partial = 0;
#endif
_Block * g_initial_block = NULL;


#ifdef SCALAR
void nqueens_expand_bf(_BlockStack* _stack, int* _depth, int* num);
#else
void nqueens_expand_bf(_BlockStack* _stack, int* _depth, int* num, int n);
#endif




#ifdef SCALAR
/***************************************************************\
*****************************************************************
*******                                                    ******
*******                 SCALAR SPECIFIC CODE               ******
*******                                                    ******
*****************************************************************                    
\****************************************************************/


int ok(char n, char *a) {
  for (int i = 0; i < n; i++) {
    char p = a[i];

    for (int j = i + 1; j < n; j++) {
      char q = a[j];
      if (q == p || q == p - (j - i) || q == p + (j - i))
        return 0;
    }
  }
  return 1;
}




/*Pseudo tail recursive nqueens matching our language spec*/
void nqueens(char n, char j, char *a, int *num, int _callIndex) {
#ifdef BLOCK_PROFILE
  profiler->record_single();
#endif

  if (_callIndex != -1) {
    a[j - 1] = _callIndex;
    if (!ok(j, a)) {
#ifdef PARALLELISM_PROFILE
      parallelismProfiler->recordNonBlockedTruncate();
#endif
      return;
    }
  }

  if (n == j) {
    *num += 1;
#ifdef PARALLELISM_PROFILE
    parallelismProfiler->recordNonBlockedTruncate();
#endif
    return;
  }

#ifdef PARALLELISM_PROFILE
  parallelismProfiler->recordNonBlockedRecurse();
#endif

  /* try each possible position for queen <j> */
  for (int i = 0; i < n; i++) {
    nqueens(n, j + 1, a, num, i);
  }
}

void nqueens_expand_bf(_BlockStack* _stack, int* _depth, int* num);


#ifdef REEXP
/*Depth First execution of i-th children to limit the memory consumption*/
int nqueens_block(_BlockStack *_stack, int _depth, int* num, int _callIndex) {
#ifdef TRACK_TRAVERSALS
  work++;
#endif
  class _BlockSet *_set = _stack ->  get (_depth);
  class _Block *_block = _set -> block;
  class _Block *_nextBlock0 = &_set -> _BlockSet::nextBlock0;
  _nextBlock0 ->  recycle ();

  int _block_size = _block->size;
  if (_block_size <= _expandSize / expand_condition){//Do dynamic reexpansion
    dynamic_reexpand_count++;
    g_is_partial = 1;
    nqueens_expand_bf(_stack, &_depth, num);
    return 1;
  } else {
#ifdef BLOCK_PROFILE
    profiler->record(_block->size, _depth);
#endif

    for (int _bi = 0; _bi < _block->size; _bi++) {
      class _Point &_point = _block ->  get (_bi);
      char *a = _point.b;

      //if (_callIndex != -1) {  // this check not necessary as block is done after expansion
      a[_depth] = _callIndex;
      if (!ok(_depth + 1, a)) {
#ifdef PARALLELISM_PROFILE
        parallelismProfiler->recordTruncate();
#endif
        continue;
      }
      //}

      if (g_nqueens == _depth + 1) {
#ifdef PARALLELISM_PROFILE
        parallelismProfiler->recordTruncate();
#endif
        *num += 1;
        continue;
      }

#ifdef PARALLELISM_PROFILE
      parallelismProfiler->recordRecurse();
#endif
      /* try each possible position for queen <j> */
      _nextBlock0->add(a, _depth + 1);
    }

    if (_nextBlock0 -> _Block::size > 0) {
      _stack ->  get (_depth + 1) -> _BlockSet::block = _nextBlock0;
#ifdef BLOCK_PROFILE
#ifdef EXPAND_PROFILE
      profiler->record_bef_exp_size(_depth + 1, _nextBlock0->size);
      profiler->record_aft_exp_size(_depth + 1, _nextBlock0->size);
#ifdef INCLUSIVE
      profiler->record_w_wo_exp_ratio(_depth + 1, 1);	
#endif
#endif
#endif
      int skip = 0;
      for (int i = 0; i < g_nqueens; i++) {
        skip =  nqueens_block(_stack, _depth + 1, num, i);
        if (skip) break;
      }
    }
  }
#ifdef PARALLELISM_PROFILE
  parallelismProfiler->blockEnd();
#endif
  return 0;
}
#endif
#ifdef NOREEXP

/*Depth First execution of i-th children to limit the memory consumption*/
void nqueens_block(_BlockStack *_stack, int _depth, int* num, int _callIndex) {
#ifdef TRACK_TRAVERSALS
  work++;
#endif
  class _BlockSet *_set = _stack ->  get (_depth);
  class _Block *_block = _set -> block;
  class _Block *_nextBlock0 = &_set -> _BlockSet::nextBlock0;
  _nextBlock0 ->  recycle ();
#ifdef BLOCK_PROFILE
  profiler.record(_block->size, _depth);
#endif

  for (int _bi = 0; _bi < _block->size; _bi++) {
    class _Point &_point = _block ->  get (_bi);
    char *a = _point.b;

    //if (_callIndex != -1) {  // this check not necessary as block is done after expansion
    a[_depth] = _callIndex;
    if (!ok(_depth + 1, a)) {
#ifdef PARALLELISM_PROFILE
      parallelismProfiler->recordTruncate();
#endif
      continue;
    }
    //}

    if (g_nqueens == _depth + 1) {
#ifdef PARALLELISM_PROFILE
      parallelismProfiler->recordTruncate();
#endif
      *num += 1;
      continue;
    }

#ifdef PARALLELISM_PROFILE
    parallelismProfiler->recordRecurse();
#endif
    /* try each possible position for queen <j> */
    _nextBlock0->add(a, _depth + 1);
  }

  if (_nextBlock0 -> _Block::size > 0) {
    _stack ->  get (_depth + 1) -> _BlockSet::block = _nextBlock0;
    for (int i = 0; i < g_nqueens; i++) {
      nqueens_block(_stack, _depth + 1, num, i);
    }
  }
#ifdef PARALLELISM_PROFILE
  parallelismProfiler->blockEnd();
#endif
}

#endif

/*Breadth First execution to expand the number of tasks in software block*/
void nqueens_expand_bf(_BlockStack* _stack, int* _depth, int* num){
#ifdef TRACK_TRAVERSALS
  work++;
#endif
  class _BlockSet *_set = _stack ->  get (*_depth);
  class _Block *_block = _set -> block;
  class _Block *_nextBlock0 = &_set -> _BlockSet::nextBlock0;
  _nextBlock0 ->  recycle ();

#ifdef BLOCK_PROFILE 
  profiler->record(_block->size, *_depth);

#ifdef REEXP
#ifdef EXPAND_PROFILE
  if (dynamic_reexpand_count) 
  {
    profiler->record_reexpansion(*_depth);
  }

  int c_bef_exp_size[g_nqueens];
  for (int i = 0; i < g_nqueens; ++i){
    c_bef_exp_size[i] = 0;
  }
  int max_c_bef_exp_size = 0;
#endif
#endif
#endif

  if (g_nqueens == *_depth) {
#ifdef PARALLELISM_PROFILE
    for (int pi = 0; pi < _block->size; _bi++)
      parallelismProfiler->recordTruncate();
#endif
    *num += _block->size;
  } else {
    for(int i = 0; i < g_nqueens; ++i){
      for (int _bi = 0; _bi < _block->size; _bi++) {
        class _Point &_point = _block ->  get (_bi);
        char *a = _point.b;

        a[*_depth] = i;
        if (!ok(*_depth + 1, a)) {
#ifdef PARALLELISM_PROFILE
          parallelismProfiler->recordTruncate();
#endif
          continue;
        }

#ifdef PARALLELISM_PROFILE
        parallelismProfiler->recordRecurse();
#endif
        _nextBlock0->add(a, *_depth + 1);
      }
#ifdef REEXP
#ifdef BLOCK_PROFILE
#ifdef EXPAND_PROFILE
      c_bef_exp_size[i] = _nextBlock0->size;
      for (int j = i - 1; j >= 0; --j){
        c_bef_exp_size[i] -= c_bef_exp_size[j];
      }
      max_c_bef_exp_size = max(max_c_bef_exp_size, c_bef_exp_size[i]);
#endif
#endif
#endif
    }
  }
  
  #ifdef NOREEXP
  //Free old stack space
  if (!*_depth){
    delete g_initial_block;
  } else {
    _stack->release(*_depth - 1);
  }
  #endif

  int _nextblock0_size = _nextBlock0 -> _Block::size;

#ifdef _DEBUG
  cout << "This is _nextblock0_size: " << _nextblock0_size << endl;
  for (int j = 0; j < _nextblock0_size; ++j){
    for (int k = 0; k < g_nqueens; ++k){
      printf("%d ", (int)_nextBlock0->points[j].b[k]);
    }
    cout << endl;
  }
  cout << endl;
#endif

#ifdef REEXP
  //Free old stack space
  if (!g_is_partial){
    if (!*_depth){
      delete g_initial_block;
    } else
    {
      _stack->release(*_depth - 1);
    }
  }
#endif

  *_depth += 1;

#ifdef REEXP
#ifdef BLOCK_PROFILE
#ifdef EXPAND_PROFILE
  if (dynamic_reexpand_count == 0){
    if(_nextblock0_size) {
      profiler->record_bef_exp_size(*_depth, _nextblock0_size);
#ifdef INCLUSIVE
      profiler->record_w_wo_exp_ratio(*_depth, 1);	
#endif
    }
  } else{
    for (int i = 0; i < g_nqueens; ++i){
      if (c_bef_exp_size[i]) profiler->record_bef_exp_size(*_depth, c_bef_exp_size[i]);
    }
    if(_nextblock0_size) profiler->record_w_wo_exp_ratio(*_depth, _nextblock0_size / (double)max_c_bef_exp_size);	
  }
  if(_nextblock0_size) {
    profiler->record_aft_exp_size(*_depth, _nextblock0_size);
  }
#endif
#endif
#endif

#ifdef REEXP
  if (_nextblock0_size > 0 && _nextblock0_size <= _expandSize / expand_condition) {
    _stack ->  get (*_depth) -> _BlockSet::block = _nextBlock0;
    nqueens_expand_bf(_stack, _depth, num);
  } else { //Reach the buffer size, or finish all evaluation
    if (!dynamic_reexpand_count){// only print for the first time
      cout << "This is the max block buffer size for dfs: " << _nextblock0_size << endl;
      cout << "This is the result now: " << *num << endl;
    }

    if (_nextblock0_size){
      _stack ->  get (*_depth) -> _BlockSet::block = _nextBlock0;
      for (int i = 0; i < g_nqueens; i++) {
        nqueens_block(_stack, *_depth, num, i);
      }
    }
  }
#ifdef PARALLELISM_PROFILE
  parallelismProfiler->blockEnd();
#endif
}
#endif
#ifdef NOREEXP

  if (_nextblock0_size > 0 && _nextblock0_size <= _expandSize / g_nqueens) {
    _stack ->  get (*_depth) -> _BlockSet::block = _nextBlock0;
    nqueens_expand_bf(_stack, _depth, num);
  } else { //Reach the buffer size, or finish all evaluation
    _stack ->  get (*_depth) -> _BlockSet::block = _nextBlock0;
  }
#ifdef PARALLELISM_PROFILE
  parallelismProfiler->blockEnd();
#endif

#endif


#endif
#ifndef SCALAR
/***************************************************************\
*****************************************************************
*******                                                    ******
*******                 SIMD SPECIFIC CODE                 ******
*******                                                    ******
*****************************************************************                    
\****************************************************************/

const int MY_SIMD_WIDTH = LOCAL_SIMD_WIDTH; //16;


int ok(char n, char *a) {
  for (int i = 0; i < n; i++) {
    char p = a[i];

    for (int j = i + 1; j < n; j++) {
      char q = a[j];
      if (q == p || q == p - (j - i) || q == p + (j - i))
        return 0;
    }
  }
  return 1;
}


/*Pseudo tail recursive nqueens matching our language spec*/
void nqueens(char n, char j, char *a, int *num, int _callIndex) {
/*#ifdef BLOCK_PROFILE
  profiler.record_single();
#endif*/

  if (_callIndex != -1) {
    a[j - 1] = _callIndex;
    if (!ok(j, a)) return;
  }
  if (n == j) {
    *num += 1;
    return;
  }

  /* try each possible position for queen <j> */
  for (int i = 0; i < n; i++) {
    nqueens(n, j + 1, a, num, i);
  }
}

/*sequential check for blocked code*/
int ok(char n, _Block *_block, int _bi) {
  for (int i = 0; i < n; i++) {
    char p = _block->get(_bi, i);
    for (int j = i + 1; j < n; j++) {
      char q = _block->get(_bi, j);
      if (q == p || q == p - (j - i) || q == p + (j - i))
        return 0;
    }
  }
  return 1;
}


/*sequential processing for Depth First Execution*/
inline void process_point(_Block *_block, _Block *_nextBlock0, int _bi, char n, char j, int _callIndex, int *num) {
  //if (_callIndex != -1) {  // this check not necessary as block is done after expansion
  _block->set(_bi, j - 1, _callIndex);
  if (!ok(j, _block, _bi)) return;
  //}

  if (n == j) {
    *num += 1;
    return;
  }

  /* try each possible position for queen <j> */
  _nextBlock0->add(_block, _bi, j);
}

/*sequential processing for Breadth First Execution*/
inline void process_point_bf(_Block *_block, _Block *_nextBlock0, int _bi, char n, char j, int _callIndex, int *num) {
  _block->set(_bi, j - 1, _callIndex);
  if (!ok(j, _block, _bi)) return;

  _nextBlock0->add(_block, _bi, j);
}



/*simd check for blocked code*/
void ok_vec(char n, _Block *_block, int _si, int *ret_mask) {
  for (int i = 0; i < n; i++) {
    _MM_TYPE_I vec_p = _MM_LOADU_I((_MM_TYPE_I*)_block->getptr(_si, i));
    for (int j = i + 1; j < n; j++) {
      _MM_TYPE_I vec_q = _MM_LOADU_I((_MM_TYPE_I*)_block->getptr(_si, j));
      _MM_TYPE_I vec_cond1 = _MM_CMPEQ_SIG8(vec_q, vec_p);
      _MM_TYPE_I vec_j_sub_i = _MM_SET_8(j - i);
      _MM_TYPE_I vec_cond2 = _MM_CMPEQ_SIG8(vec_q, _MM_SUB_8(vec_p, vec_j_sub_i));
      _MM_TYPE_I vec_cond3 = _MM_CMPEQ_SIG8(vec_q, _MM_ADD_8(vec_p, vec_j_sub_i));
      _MM_TYPE_I vec_cond = _MM_OR_SIG(_MM_OR_SIG(vec_cond1, vec_cond2), vec_cond3);
      int mask = _MM_MOVEMASK_I(vec_cond);
      *ret_mask = *ret_mask | mask;
      if (*ret_mask == 0xffff) return;
    }
  }
}


/*simd processing for Depth First Execution, we store n in char in the block*/
inline void process_simd(_Block *_block, _Block *_nextBlock0, int _si, char n, char j, int _callIndex, int *num) {
  
  /* CDF START */
  //  Right before the section of the code we want to measure
#ifdef ENABLE_PARSEC_HOOKS
//  __parsec_roi_begin();
#endif
  /* CDF END */
  
  _MM_TYPE_I vec_callIndex = _MM_SET_8(_callIndex);
  char *dest = _block->getptr(_si, j - 1);
  _MM_STOREU_I((_MM_TYPE_I*)dest, vec_callIndex);
  int ret_mask = 0;
  ok_vec(j, _block, _si, &ret_mask);
  if (ret_mask == 0xffff) return;
  int ok_mask = ~ret_mask;

#ifdef NOSC
  if (n == j) {
    for (int i = 0; i < MY_SIMD_WIDTH; i++) {
      int f = 1 << i;
      if (f & ok_mask) *num += 1;
    }
  } else {
    for (int i = 0; i < MY_SIMD_WIDTH; i++) {
      int f = 1 << i;
      if (f & ok_mask) _nextBlock0->add(_block, _si + i, j);
    }
  }

/*#else//Streaming Compaction
  if (n == j){
    *num += g_advanceNextPtrCounts[ret_mask & 0x000000FF] 
        + g_advanceNextPtrCounts[(ret_mask & 0x0000FF00) >> 8];
  }else{
    __attribute__((aligned(16))) unsigned char tmp[16];
    unsigned index = 0;
    //do first 8
    *((__int64*)tmp) = g_shuffletable[ret_mask & 0x000000FF];
    index += g_advanceNextPtrCounts[ret_mask & 0x000000FF];
    // now second 8
    *((__int64*)&tmp[index]) = 0x0808080808080808 + g_shuffletable[(ret_mask & 0x0000FF00) >> 8];
    index += g_advanceNextPtrCounts[(ret_mask & 0x0000FF00) >> 8];
    // fill rest with 0xFF
    memset(&tmp[index], 0xFF, 16 - index);

    _MM_TYPE_I vec_shuffleTable =  _MM_LOAD_SIG((const _MM_TYPE_I *) tmp);

    for (int i = 0; i < j; i++){
      _MM_TYPE_I vec_n = _MM_LOADU_I((_MM_TYPE_I*)_block->getptr(_si, i));
      vec_n = _MM_SHUFFLE_8(vec_n, vec_shuffleTable);
      _MM_STOREU_I((_MM_TYPE_I*)&_nextBlock0->a[i * _Block::max_block + _nextBlock0->size], vec_n);

    }

    _nextBlock0->size += index;

  }*/

#endif

/* CDF START */
// When we are done measuring
#ifdef ENABLE_PARSEC_HOOKS
//  __parsec_roi_end();
#endif
/* CDF END */


}

#ifdef REEXP
/*Depth First execution of i-th children to limit the memory consumption*/
int nqueens_block(_BlockStack *_stack, int _depth, int* num, int _callIndex, int n) {
  class _BlockSet *_set = _stack ->  get (_depth);
  class _Block *_block = _set -> block;
  class _Block *_nextBlock0 = &_set -> _BlockSet::nextBlock0;
  _nextBlock0 ->  recycle ();
/*#ifdef BLOCK_PROFILE
  profiler.record(_block->size, _depth);
#endif*/
  int _block_size = _block->size;
  if (_block_size <= _expandSize / expand_condition){//Do dynamic reexpansion
    dynamic_reexpand_count++;
    g_is_partial = 1;
    nqueens_expand_bf(_stack, &_depth, num, n);
    return 1;
  } else {
    int _si = 0;
    for (; _si < (_block->size - MY_SIMD_WIDTH + 1); _si += MY_SIMD_WIDTH) {
      process_simd(_block, _nextBlock0, _si, n, _depth+1, _callIndex, num);
    }

    //Process the rest
    for (int _bi = _si; _bi < _block->size; _bi++) {
      process_point(_block, _nextBlock0, _bi, n, _depth+1, _callIndex, num);

    }   

    if (_nextBlock0 -> _Block::size > 0) {
      _stack ->  get (_depth + 1) -> _BlockSet::block = _nextBlock0;
      int skip = 0;
      for (int i = 0; i < n; i++) {
        skip = nqueens_block(_stack, _depth + 1, num, i, n);
        if (skip) break;
      }
    }
  }

  return 0;
}
#endif
#ifdef NOREEXP
/*Depth First execution of i-th children to limit the memory consumption*/
void nqueens_block(_BlockStack *_stack, int _depth, int* num, int _callIndex, int n) {
  class _BlockSet *_set = _stack ->  get (_depth);
  class _Block *_block = _set -> block;
  class _Block *_nextBlock0 = &_set -> _BlockSet::nextBlock0;
  _nextBlock0 ->  recycle ();
#ifdef BLOCK_PROFILE
  profiler.record(_block->size, _depth);
#endif

  int _si = 0;
#ifdef NOSIMD
#else
  for (; _si < (_block->size - MY_SIMD_WIDTH + 1); _si += MY_SIMD_WIDTH) {
    process_simd(_block, _nextBlock0, _si, n, _depth+1, _callIndex, num);
  }
#endif
  //Process the rest
  for (int _bi = _si; _bi < _block->size; _bi++) {
    process_point(_block, _nextBlock0, _bi, n, _depth+1, _callIndex, num);

  }
  if (_nextBlock0 -> _Block::size > 0) {
    _stack ->  get (_depth + 1) -> _BlockSet::block = _nextBlock0;
    for (int i = 0; i < n; i++) {
      nqueens_block(_stack, _depth + 1, num, i, n);
    }
  }
}
#endif

/*simd processing for Breadth First Execution, we store n in char in the block*/
inline void process_simd_bf(_Block *_block, _Block *_nextBlock0, int _si, char n, char j, int _callIndex, int *num) {
  
  /* CDF START */
  //  Right before the section of the code we want to measure
#ifdef ENABLE_PARSEC_HOOKS
//  __parsec_roi_begin();
#endif
  /* CDF END */
    
  _MM_TYPE_I vec_callIndex = _MM_SET_8(_callIndex);
  char *dest = _block->getptr(_si, j - 1);
  _MM_STOREU_I((_MM_TYPE_I*)dest, vec_callIndex);
  int ret_mask = 0;
  ok_vec(j, _block, _si, &ret_mask);
  if (ret_mask == 0xffff) return;
  int ok_mask = ~ret_mask;


  for (int i = 0; i < MY_SIMD_WIDTH; i++) {
    int f = 1 << i;
    if (f & ok_mask) _nextBlock0->add(_block, _si + i, j);
  }
  
  /* CDF START */
  // When we are done measuring
#ifdef ENABLE_PARSEC_HOOKS
//  __parsec_roi_end();
#endif
  /* CDF END */
  
}

/*Breadth First execution to expand the number of tasks in software block*/
void nqueens_expand_bf(_BlockStack* _stack, int* _depth, int* num, int n){
  class _BlockSet *_set = _stack ->  get (*_depth);
  class _Block *_block = _set -> block;
  class _Block *_nextBlock0 = &_set -> _BlockSet::nextBlock0;
  _nextBlock0 ->  recycle ();
#ifdef BLOCK_PROFILE
  profiler.record(_block->size, *_depth);
#endif

  if (n == *_depth) {
    *num += _block->size;
  } else {
    for(int i = 0; i < n; ++i){
      int _si = 0;
      for (; _si < (_block->size - MY_SIMD_WIDTH + 1); _si += MY_SIMD_WIDTH) {
        process_simd_bf(_block, _nextBlock0, _si, n, *_depth+1, i, num);
      }

      //Process the rest
      for (int _bi = _si; _bi < _block->size; _bi++) {
        process_point_bf(_block, _nextBlock0, _bi, n, *_depth+1, i, num);

      }
    }
  }
#ifdef REEXP
  //Free old stack space
  if (!g_is_partial){
    if (!*_depth){
      delete g_initial_block;
    } else
    {
      _stack->release(*_depth - 1);
    }
  }
#endif
#ifdef NOREEXP  
  //Free old stack space
  if (!*_depth){
    delete g_initial_block;
  } else
  {
    _stack->release(*_depth - 1);
  }
#endif

  int _nextblock0_size = _nextBlock0 -> _Block::size;
#ifdef _DEBUG
  cout << "This is max_block: " << _Block::max_block << endl;
  cout << "This is _nextblock0_size: " << _nextblock0_size << endl;
  for (int j = 0; j < _nextblock0_size; ++j){
    for (int k = 0; k < n; ++k){
      printf("%d ", (int)_nextBlock0->a[k * _Block::max_block + j]);
    }
    cout << endl;
  }
  cout << endl;
#endif

  *_depth += 1;
#ifdef REEXP
  if (_nextblock0_size > 0 && _nextblock0_size <= _expandSize / expand_condition) {
    _stack ->  get (*_depth) -> _BlockSet::block = _nextBlock0;
    nqueens_expand_bf(_stack, _depth, num, n);
  } else { //Reach the buffer size, or finish all evaluation
    if (!dynamic_reexpand_count){// only print for the first time
      cout << "This is the max block buffer size for dfs: " << _nextblock0_size << endl;
      cout << "This is the result now: " << *num << endl;
    }

    if (_nextblock0_size){
      _stack ->  get (*_depth) -> _BlockSet::block = _nextBlock0;
      for (int i = 0; i < n; i++) {
        nqueens_block(_stack, *_depth, num, i, n);
      }
    }
  }
#endif
#ifdef NOREEXP
  if (_nextblock0_size > 0 && _nextblock0_size <= _expandSize / n ) {
    _stack ->  get (*_depth) -> _BlockSet::block = _nextBlock0;
    nqueens_expand_bf(_stack, _depth, num, n);
  } else { //Reach the buffer size, or finish all evaluation
    _stack ->  get (*_depth) -> _BlockSet::block = _nextBlock0;
  }
#endif
}

#endif






/*******************************************************************/








#ifndef SCALAR
/*Benchmark entrance called by harness*/
int app_main(int argc, char **argv) {

/* CDF START */
// When the benchmark begins (main), use the same string "__parsec_streamcluster", it's just ignored
#ifdef ENABLE_PARSEC_HOOKS
	 __parsec_bench_begin(__parsec_streamcluster);
#endif
/* CDF END */
  
  if (argc < 1) {
    printf("number of queens required\n");
    return 1;
  }
  if (argc > 2)
    printf("extra arguments being ignored\n");

  int n = atoi(argv[0]);
  printf("running queens %d\n", n);

  if (argc == 2) _expandSize = pow(2.0, atoi(argv[1]));


  Harness::start_timing();
  //_expandDepth = Harness::get_splice_depth();

  char *a = (char *)alloca(n * sizeof(char));
  int num = 0;
  
#ifdef REEXP
  expand_condition_table = (int*)malloc(25 * sizeof(int));
  //Specific for n = 13
  for (int i = 0; i < 25; ++i){
    if (i < 7) expand_condition_table[i] = 13;
    else if (i < 8) expand_condition_table[i] = 11;
    else if (i < 12) expand_condition_table[i] = 8;
    else if (i < 17) expand_condition_table[i] = 4;
    else if (i < 23) expand_condition_table[i] = 4;
    else expand_condition_table[i] = 2;
  }
  expand_condition = n;
  if (argc == 2 && n == 13)	expand_condition = expand_condition_table[atoi(argv[1])];
#endif

  //Initialize software block stack
  _Block::n = n;
  cout << "Set fixed max block buffer size, _expandSize: " << _expandSize << endl;
  _Block::max_block = _expandSize;
  Harness::set_block_size(_expandSize);
  class _BlockStack * _stack = new _BlockStack;
  class _Block * _block = new _Block;
  g_initial_block = _block;

  _block->add(a, 0);
  int _depth = 0;
  _stack->get (_depth) -> block = _block;

#ifdef REEXP
  //Start to execute blocked nqueens 
  if (_expandSize >= n) nqueens_expand_bf(_stack, &_depth, &num, n);
  else{ int df_block_size = _stack->get(_depth)->block->size;
    cout << "This is the max block buffer size for dfs: " << df_block_size << endl;
    cout << "This is the result now: " << num << endl;

    if (df_block_size){
      for (int i = 0; i < n; i++) {
        nqueens_block(_stack, _depth, &num, i, n);
      }
    }
  }

  delete _stack;
  if (_expandSize < n) delete _block;
  Harness::stop_timing();
#ifdef BLOCK_PROFILE
  profiler.output();
#endif

  printf("Solutions for %d nqueens = %d\n", n, num);

#ifdef PROFILE_SPACE_USE
  cout << "This is max space use (Bytes): " << m_space << endl;
  cout << "This is the total number of new operations for block: " << total_malloc << endl;
#endif

#endif
#ifdef NOREEXP
  //Start to execute blocked nqueens 
  if (_expandSize >= n) nqueens_expand_bf(_stack, &_depth, &num, n);
  int df_block_size = _stack->get(_depth)->block->size;
  cout << "This is the max block buffer size for dfs: " << df_block_size << endl;
  cout << "This is the result now: " << num << endl;

  if (df_block_size){
    for (int i = 0; i < n; i++) {
      nqueens_block(_stack, _depth, &num, i, n);
    }
  }

  delete _stack;
  if (_expandSize < n) delete _block;

  Harness::stop_timing();
#ifdef BLOCK_PROFILE
  profiler.output();
#endif

  printf("nqueens = %d\n", num);

#ifdef PROFILE_SPACE_USE
  cout << "This is max space use (Bytes): " << m_space << endl;
  cout << "This is total malloc counts: " << total_malloc << endl;
#endif

#endif

/* CDF START */
  // Before you exit the application
#ifdef ENABLE_PARSEC_HOOKS
  __parsec_bench_end(); 
#endif
/* CDF END */

  return 0;
}
#endif
#ifdef SCALAR

/*Benchmark entrance called by harness*/
int app_main(int argc, char **argv) {
  if (argc < 1) {
    printf("number of queens required\n");
    return 1;
  }
#ifdef REEXP
  if (argc > 3)
    printf("extra arguments being ignored\n");
#else
  if (argc > 2)
    printf("extra arguments being ignored\n");
#endif

#ifdef REEXP
  int vec_width = 16;
#endif
  g_nqueens = atoi(argv[0]);
  printf("running queens %d\n", g_nqueens);

  if (argc >= 2) _expandSize = pow(2.0, atoi(argv[1]));
#ifdef REEXP
  if (argc >= 3)  vec_width = atoi(argv[2]);
#endif

#ifdef PARALLELISM_PROFILE
  parallelismProfiler = new ParallelismProfiler;
#endif

#ifdef REEXP
#ifdef BLOCK_PROFILE
  cout << "The vector width is " << vec_width << endl;
  profiler = new BlockProfiler(vec_width);
#endif
#endif
  Harness::start_timing();
  //_expandDepth = Harness::get_splice_depth();

  char *a = (char *)alloca(g_nqueens * sizeof(char));
  int num = 0;

#ifdef REEXP
  //Specific for n = 10
  expand_condition_table_10 = (int*)malloc(25 * sizeof(int));
  for (int i = 0; i < 25; ++i){
    if (i < 4) expand_condition_table_10[i] = 10;
    else if (i < 8 ) expand_condition_table_10[i] = 8;
    else if (i < 12) expand_condition_table_10[i] = 4;
    else expand_condition_table_10[i] = 2;
  }

  //Specific for n = 11
  expand_condition_table_11 = (int*)malloc(25 * sizeof(int));
  for (int i = 0; i < 25; ++i){
    if (i < 4) expand_condition_table_11[i] = 11;
    else if (i < 10 ) expand_condition_table_11[i] = 8;
    else if (i < 14) expand_condition_table_11[i] = 4;
    else expand_condition_table_11[i] = 2;
  }

  //Specific for n = 12
  expand_condition_table_12 = (int*)malloc(25 * sizeof(int));
  for (int i = 0; i < 25; ++i){
    if (i < 4) expand_condition_table_12[i] = 12;
    else if (i < 15) expand_condition_table_12[i] = 8;
    else if (i < 17) expand_condition_table_12[i] = 4;
    else expand_condition_table_12[i] = 2;
  }

  //Specific for n = 13
  expand_condition_table_13 = (int*)malloc(25 * sizeof(int));
  for (int i = 0; i < 25; ++i){
    if (i < 7) expand_condition_table_13[i] = 13;
    else if (i < 8) expand_condition_table_13[i] = 11;
    else if (i < 12) expand_condition_table_13[i] = 8;
    else if (i < 23) expand_condition_table_13[i] = 4;
    else expand_condition_table_13[i] = 2;
  }

  //Specific for n = 14
  expand_condition_table_14 = (int*)malloc(30 * sizeof(int));
  for (int i = 0; i < 30; ++i){
    if (i < 7) expand_condition_table_14[i] = 14;
    else if (i < 13) expand_condition_table_14[i] = 11;
    else if (i < 15) expand_condition_table_14[i] = 8;
    else if (i < 23) expand_condition_table_14[i] = 4;
    else expand_condition_table_14[i] = 2;
  }


  expand_condition = g_nqueens;
  if (argc >= 2 && g_nqueens == 10) expand_condition = expand_condition_table_10[atoi(argv[1])];
  if (argc >= 2 && g_nqueens == 11) expand_condition = expand_condition_table_11[atoi(argv[1])];
  if (argc >= 2 && g_nqueens == 12) expand_condition = expand_condition_table_12[atoi(argv[1])];
  if (argc >= 2 && g_nqueens == 13) expand_condition = expand_condition_table_13[atoi(argv[1])];
  if (argc >= 2 && g_nqueens == 14) expand_condition = expand_condition_table_14[atoi(argv[1])];
  
#endif

  //Initialize software block stack
  cout << "Set fixed max block buffer size, _expandSize: " << _expandSize << endl;
  _Block::max_block = _expandSize;
  Harness::set_block_size(_expandSize);
  class _BlockStack * _stack = new _BlockStack;
  class _Block * _block = new _Block;
  g_initial_block = _block;

  _block->add(a);
  int _depth = 0;
  _stack->get (_depth) -> block = _block;

  //Start to execute blocked nqueens 
#ifdef REEXP
  if (_expandSize >= g_nqueens){
#ifdef BLOCK_PROFILE
#ifdef EXPAND_PROFILE
    profiler->record_bef_exp_size(0, 1);
    profiler->record_aft_exp_size(0, 1);
#ifdef INCLUSIVE
    profiler->record_w_wo_exp_ratio(0, 1);
#endif
#endif
#endif
    nqueens_expand_bf(_stack, &_depth, &num);
  }
  else{
    int df_block_size = _stack->get(_depth)->block->size;
    cout << "This is the max block buffer size for dfs: " << df_block_size << endl;
    cout << "This is the result now: " << num << endl;

    if (df_block_size){
      for (int i = 0; i < g_nqueens; i++) {
        nqueens_block(_stack, _depth, &num, i);
      }
    }
  }

  delete _stack;
  if (_expandSize < g_nqueens) delete _block;

  Harness::stop_timing();
#ifdef BLOCK_PROFILE
  profiler->output();
  //    profiler.outputBlockInfo();//For output task distribution profile data
#ifdef EXPAND_PROFILE
  profiler->outputReexpandInfo();
#endif
  delete profiler;
#endif
#ifdef PARALLELISM_PROFILE
  parallelismProfiler->output();
  delete parallelismProfiler;
#endif

#ifdef TRACK_TRAVERSALS
  cout << "work: " << work << endl;
#endif

  printf("nqueens = %d\n", num);

  printf("This is dynamic reexpand counts: %d\n", dynamic_reexpand_count);

#ifdef PROFILE_SPACE_USE
  cout << "This is max space use (Bytes): " << m_space << endl;
  cout << "This is the total number of new operations for block: " << total_malloc << endl;
#endif
#endif
#ifdef NOREEXP

  //Start to execute blocked nqueens 
  if (_expandSize >= g_nqueens)  nqueens_expand_bf(_stack, &_depth, &num);
  int df_block_size = _stack->get(_depth)->block->size;
  cout << "This is the max block buffer size for dfs: " << df_block_size << endl;
  cout << "This is the result now: " << num << endl;

  if (df_block_size){
    for (int i = 0; i < g_nqueens; i++) {
      nqueens_block(_stack, _depth, &num, i);
    }
  }

  delete _stack;
  if (_expandSize < g_nqueens) delete _block;
  Harness::stop_timing();

#ifdef BLOCK_PROFILE
  profiler.output();
#ifdef BLOCKINFO
  profiler.outputBlockInfo();//For output task distribution profile data 
#endif 
#endif

#ifdef TRACK_TRAVERSALS
  cout << "work: " << work << endl;
#endif

  printf("nqueens = %d\n", num);

#ifdef PROFILE_SPACE_USE
  cout << "This is max space use (Bytes): " << m_space << endl;
  cout << "This is total malloc counts: " << total_malloc << endl;
#endif

#endif

  return 0;
}


#endif