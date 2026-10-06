// Harness written for this package (MIT licence, like the rest of scripts/): one run of
// Tanabe's SHADE 1.0.1 C++ code on one CEC2014 function.  It is compiled together with
// that release's de.h, search_algorithm.cc and shade.cc and with the CEC2014 source
// cec14_test_func.cc, none of which is redistributed here; run_shade101.py fetches
// nothing, checks their sha256 and applies the two edits it documents.
// Parameters as in the release's de_test.cc: N = 100, H = 100, |A| = N, 10,000*D
// evaluations.  Usage: shade101 func dim run seed   (seed is passed to glibc srand)
#include "de.h"
double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag,*SS;
int g_function_number; int g_problem_size; unsigned int g_max_num_evaluations;
int g_pop_size; int g_memory_size; int g_arc_size;
int main(int argc, char **argv) {
  if (argc != 5) { cerr << "usage: shade101 func dim run seed" << endl; return 2; }
  g_function_number = atoi(argv[1]); g_problem_size = atoi(argv[2]);
  int run = atoi(argv[3]); srand((unsigned)strtoul(argv[4], 0, 10));
  g_max_num_evaluations = g_problem_size * 10000;
  g_pop_size = 100; g_memory_size = 100; g_arc_size = g_pop_size;
  searchAlgorithm *alg = new SHADE();
  Fitness e = alg->run();
  cout << setprecision(17) << g_function_number << " " << g_problem_size << " " << run << " " << e << endl;
  return 0;
}
