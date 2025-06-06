#include "cantera/core.h"
#include "cantera/numerics/Integrator.h"
#include <iostream>

extern "C"
{
  void myfgh(int need[], int &nx, double x[], int &nf, int &nh, int iusr[], double rusr[], double f[], double g[], double h[]);
	   
  void mymix(int &nx, double x1[], double x2[], double alpha[], int iusr[], double rusr[] );
  
  void toxhat(double x[], double ptcl[], int &nx, double rusr[] );
  
  void myfnn(int &nx, double x[], double fnn[]);
  
  void fromxhat(double ptcl[], double x[], int &nx, double rusr[] );
} //fortran90/C interface

using namespace Cantera;

namespace Gl{  // global variables

	extern shared_ptr< Solution > sol;
	extern shared_ptr< ThermoPhase > gas;
	
	extern void initfgh();
	
	extern void initfnn();
	
	extern int ia[100];
	extern int ib[100];
	extern int n1[100];
	extern int n2[100];
	
	extern double A[1000000];
	extern double b[10000];
	
	extern int nLayers;
	extern int nNeurons;
	

}