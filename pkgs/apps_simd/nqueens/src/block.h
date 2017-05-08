/**********************************************************/
/* This code is for PLDI-15 Artifact Evaluation only      */ 
/* and will be released with further copyright information*/ 
/* File: Basic sequential software block stack            */
/**********************************************************/

#include <vector>
#include <cassert>

#define D_MAX_BLOCK_SIZE 128
//#define PROFILE_SPACE_USE

#ifdef PROFILE_SPACE_USE
long long m_space = 0;
long long c_space = 0;
long long total_malloc = 0;
#endif

int g_nqueens = 0;

using namespace std;

class _Point {
 public:
  char* b;
  _Point(){
    this->b = new char[g_nqueens]();
  }
  _Point(char* b){
    this->b = new char[g_nqueens];
    memcpy(this->b, b, g_nqueens * sizeof(char));
  }
  ~_Point(){
    delete [] b;
  }
};


class _Block {
 public:
  _Block();
  ~_Block();

  void add(char *b);
  void add(char *b, int d);
  _Point& get(int i) { return points[i]; }
  void recycle() { size = 0; }
  bool is_full() { return size == max_block; }
  bool is_empty() { return size == 0; }

  _Point* points;
  int size;

  static int max_block;
};

class _BlockSet
{
 public:
  _BlockSet() {}
  ~_BlockSet() {}

  _Block *block;
  _Block nextBlock0;
};

class _BlockStack
{
 public:
  _BlockStack() {}
  ~_BlockStack();

  _BlockSet* get(int i);
  void release(int _depth){
    delete this->items[_depth];
    this->items[_depth] = NULL;
  }
  vector<_BlockSet *> items;
};

int _Block::max_block = 128;

_Block::_Block() {
  points = new _Point[max_block];
  size = 0;

#ifdef PROFILE_SPACE_USE
  c_space += max_block * sizeof(char) * g_nqueens;
  m_space = max(m_space, c_space);
  total_malloc++;
#endif
}

_Block::~_Block() {
  delete [] points;
#ifdef PROFILE_SPACE_USE
  c_space -= max_block * sizeof(char) * g_nqueens;
  m_space = max(m_space, c_space);
#endif
}

void _Block::add(char *b){
  assert(size < max_block);
  memcpy(points[size].b, b, g_nqueens * sizeof(char));
  size++;
}

void _Block::add(char *b, int d){
  assert(size < max_block);
  memcpy(points[size].b, b, d * sizeof(char));
  size++;
}

_BlockStack::~_BlockStack() {
  for (int i = 0; i < items.size(); i++) {
    if (items[i]) delete items[i];
  }
}

_BlockSet* _BlockStack::get(int i) {
  while (i >= items.size()) {
    items.push_back(new _BlockSet());
  }
  return items[i];
}
