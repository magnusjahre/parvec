/**********************************************************/
/* This code is for PLDI-15 Artifact Evaluation only      */ 
/* and will be released with further copyright information*/ 
/* File: Sequential block w reexpansion of nqueens        */
/**********************************************************/


#include <iostream>
#include <fstream>

#include "harness.h"
#include "block.h"

/* CDF START */
//#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
/* CDF END */

#ifdef BLOCK_PROFILE
#include "blockprofiler.h"
BlockProfiler * profiler;
#endif
//Parallelism profiler, not used in our paper
//for further development
#ifdef PARALLELISM_PROFILE
#include "parallelismprofiler.h"
ParallelismProfiler *parallelismProfiler;
#endif

#ifdef TRACK_TRAVERSALS
uint64_t work = 0;
#endif

int* expand_condition_table_10;
int* expand_condition_table_11;
int* expand_condition_table_12;
int* expand_condition_table_13;
int* expand_condition_table_14;
int expand_condition;

using namespace std;

//int _expandDepth = 0;
int _expandSize = D_MAX_BLOCK_SIZE;

int dynamic_reexpand_count = 0;
_Block * g_initial_block = NULL;//For memory release
int g_is_partial = 0;

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
#ifdef BLOCK_PROFILE
#ifdef EXPAND_PROFILE
      c_bef_exp_size[i] = _nextBlock0->size;
      for (int j = i - 1; j >= 0; --j){
        c_bef_exp_size[i] -= c_bef_exp_size[j];
      }
      max_c_bef_exp_size = max(max_c_bef_exp_size, c_bef_exp_size[i]);
#endif
#endif
    }
  }

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

  //Free old stack space
  if (!g_is_partial){
    if (!*_depth){
      delete g_initial_block;
    } else
    {
      _stack->release(*_depth - 1);
    }
  }

  *_depth += 1;

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

  if (_nextblock0_size > 0 && _nextblock0_size <= _expandSize / expand_condition) {
    _stack ->  get (*_depth) -> _BlockSet::block = _nextBlock0;
    nqueens_expand_bf(_stack, _depth, num);
  } else { //Reach the buffer size, or finish all evaluation
    if (!dynamic_reexpand_count){// only print for the first time
      /*cout << "This is the max block buffer size for dfs: " << _nextblock0_size << endl;*/
      /*cout << "This is the result now: " << *num << endl;*/
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
  if (argc > 3)
    printf("extra arguments being ignored\n");

  int vec_width = 16;
  g_nqueens = atoi(argv[0]);
  printf("running queens %d\n", g_nqueens);

  if (argc >= 2) _expandSize = pow(2.0, atoi(argv[1]));
  if (argc >= 3)  vec_width = atoi(argv[2]);

#ifdef PARALLELISM_PROFILE
  parallelismProfiler = new ParallelismProfiler;
#endif

#ifdef BLOCK_PROFILE
  cout << "The vector width is " << vec_width << endl;
  profiler = new BlockProfiler(vec_width);
#endif
  Harness::start_timing();
  //_expandDepth = Harness::get_splice_depth();

  char *a = (char *)alloca(g_nqueens * sizeof(char));
  int num = 0;

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

  /* CDF START */
  //  Right before the section of the code we want to measure
#ifdef ENABLE_PARSEC_HOOKS
  __parsec_roi_begin();
#endif
  /* CDF END */

  //Start to execute blocked nqueens 
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
    /*cout << "This is the max block buffer size for dfs: " << df_block_size << endl;*/
    /*cout << "This is the result now: " << num << endl;*/

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

/* CDF START */
  //  Right before the section of the code we want to measure
#ifdef ENABLE_PARSEC_HOOKS
  __parsec_roi_end();
#endif
/* CDF END */

  printf("nqueens = %d\n", num);

  printf("This is dynamic reexpand counts: %d\n", dynamic_reexpand_count);

#ifdef PROFILE_SPACE_USE
  cout << "This is max space use (Bytes): " << m_space << endl;
  cout << "This is the total number of new operations for block: " << total_malloc << endl;
#endif

/* CDF START */
  // Before you exit the application
#ifdef ENABLE_PARSEC_HOOKS
  __parsec_bench_end();
#endif
/* CDF END */

  return 0;
}
