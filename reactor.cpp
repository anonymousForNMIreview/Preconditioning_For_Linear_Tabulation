#include "reactor.hpp"
#include <cmath>

using namespace Cantera;

namespace Gl{

	shared_ptr< Solution > sol; // = newSolution("h2o2.yaml", "ohmech", "none");
	shared_ptr< ThermoPhase > gas; // = sol->thermo();
	
	int nLayers = 6; // number of MLP layers
	int nNeurons = 30; // number of neurons in each hidden layer
	int nx = 11; // number of input/output variables
	
	int ia[100];
	int ib[100];
	int n1[100];
	int n2[100];
	
	double A[1000000];
	double b[10000]; // lines 15 to 21 are work variables for reading in the MLP weights
	
	void initfgh(){

		sol = newSolution("h2o2.yaml", "ohmech", "none"); // initialize the Cantera gas object,
		// the inputs can be modified to change the chemical mechanism		
		gas = sol->thermo();
		
	}
	
	void initfnn(){ // initialize the weights for the f^{MLP} function
		
		int i1 = 0;
		int i2 = 0; // work variables for reading in the MLP weights
		
		
		char file1[50];
		char file2[50];
		
		for (int ll = 1; ll <= nLayers; ll++){
			
			ia[ll-1] = i1;
			ib[ll-1] = i2;
			
			sprintf(file1,"./A%d.csv",ll);
			sprintf(file2,"./B%d.csv",ll);
			
			n1[ll-1] = nNeurons; n2[ll-1] = nNeurons;
			if (ll == 1){n1[ll-1] = nx;}
			if (ll == nLayers){n2[ll-1] = nx;}
		
			FILE * pFile;
			float a;
			pFile = fopen(file1,"r+");
			for (int ii=0;ii<n1[ll-1]*n2[ll-1];ii++){
				fscanf(pFile, "%f", &a);
				A[i1] = a;
				i1 = i1+1;
			}			
			fclose(pFile);
			
			pFile = fopen(file2,"r+");
			for (int ii=0;ii<n2[ll-1];ii++){
				fscanf(pFile, "%f", &a);
				b[i2] = a;
				i2 = i2+1;
			}			
			fclose(pFile);

		}
		
	}

}

using namespace Gl;

#include "custom.hpp"
// "custom.hpp" is the ReactorODEs class from lines 21-147 of the file "custom.cpp" which can be found
// at this URL: https://cantera.org/3.1/examples/cxx/custom.html (retrieved 06/05/2025)


void fromxhat(double x[], double ptcl[], int &nx, double rusr[]){
	// this function converts the normalized vector x into temperature and mass fractions for one particle
	// x[] is the input, ptcl[] is the output, nx indicates the number of dimensions of both x and ptcl
	// rusr[] are user-supplied normalization variables
	
	ptcl[0] = (x[0]*rusr[nx])+rusr[0]; // ptcl[0] is the temperature, in K
	
	for ( int ii = 1; ii < nx; ii++){
	
		ptcl[ii] = rusr[ii]*exp(-log(rusr[ii])*x[ii]) - rusr[ii]; // ptcl[ii] is the mass fraction of the ii-th
		// species in the chemical mechanism
		
	}
	
}

void toxhat(double ptcl[], double x[], int &nx, double rusr[]){
	// this function converts a particle's temperature and mass fractions into the normalized vector x
	// x[] is the input, ptcl[] is the output, nx indicates the number of dimensions of both x and ptcl
	// rusr[] are user-supplied normalization variables	
	
	x[0] = (ptcl[0]-rusr[0])/rusr[nx]; // x[0] is the normalized temperature
	
	for ( int ii = 1; ii < nx; ii++){
	
		x[ii] = - log( (ptcl[ii] + rusr[ii]) / rusr[ii] ) / log(rusr[ii]); // x[ii] is the normalized mass fraction
		// of the ii-th species in the chemical mechanism
		
	}
	
}

double fAct( double x){ // activation function of the hidden layers,
	// here a Mish function is used
	
	return x*tanh(log(1.0 + exp(x)));
	
}

void myfnn(int &nx, double x[], double fnn[]){
	// this function evaluates f^{}
	
	static int bbbb; // dummy variable used to call "initfnn" the first time "myfnn" is called
	
	double x1[100]; 
	double x2[100]; // work arrays
	
	if (bbbb!=7777){
		Gl::initfnn(); bbbb=7777;} // if "myfnn" is called for the first time, initialize the
		// f^{MLP} data structure by reading in the weights
		
	for (int ii = 0; ii < n1[0]; ii++){
		x1[ii] = x[ii]; // initialize the input
	}
	
	for (int ll = 0; ll < nLayers; ll++){
		
		for ( int kk = 0; kk < n2[ll]; kk++ ){
			x2[kk] = 0.0;
			for ( int jj = 0; jj < n1[ll]; jj++ ){
				x2[kk] += A[ ia[ll] + jj + kk*n1[ll] ]*x1[jj]; // apply weights in a dense layer
				
			}
			x2[kk] += b[ ib[ll] + kk ]; // apply the bias in a dense layer
			
			if ( ll < nLayers - 1 ){
				x2[kk] = fAct(x2[kk]); // apply the activation function in the hidden layers
			}
			
		}
		
		for ( int kk = 0; kk < n2[ll]; kk++ ){x1[kk] = x2[kk];}
				
	}
	
	for ( int kk = 0; kk < nx; kk++ ){fnn[kk] = 1.0*(x2[kk]);} // pass the output
		
	
}



// myfgh is the function passed to ISAT
void myfgh(int need[], int &nx, double x[], int &nf, int &nh, int iusr[], 
	   double rusr[], double f[], double g[], double h[])
{
	
	double Y[nx-1]; // mass fraction
	double T[1]; //temperature
	double ptcl[nx]; // particle properties
	double *solution; // Cantera solution object
	double aTol = 1e-8; //rusr[2*nx];
	double rTol = 1e-8; //rusr[2*nx+1]; //absolute and relative tolerances for the ODE integrator
	double dt = rusr[2*nx+2]; // time step over which to integrate
	double dx = rusr[2*nx+3]; // spatial increment in x for Jacobian evaluation
	double p = rusr[2*nx+4]; // user-specified pressure
	int mode = iusr[0];
	double fnn[nx]; // f^{MLP}
	
	static int aaaa; // if "myfgh" is called for the first time, initialize the
		// myfgh data structure by creating the Cantera solution object
	
	if (aaaa!=7777){
		Gl::initfgh(); aaaa=7777;} // initialize "myfgh" on the first call
		
	fromxhat(x,ptcl,nx,rusr); // convert from normalized x to particle properties
	
	T[0] = ptcl[0];
	for (int ii = 1; ii < nx; ii++){ Y[ii-1] = ptcl[ii];} // extract temperature and mass fractions		
	
	gas->setState_TPY(T[0], p, Y); // set the gas state, using the particle's temperature and mass fractions,
	// and the user-specified pressure
	
	
	ReactorODEs odes = ReactorODEs(sol); // create the ODE RHS evaluator
	
	double tnow = 0.0; // initialize time
	    
    shared_ptr<Integrator> integrator(newIntegrator("CVODE")); // create a CVODE integrator of the ODE
    
    integrator->initialize(tnow, odes); // initialize the integrator
	
	integrator->setTolerances(aTol, rTol); // set ODE integration tolerances

    integrator->integrate(dt); // integrate the chemical composition from time 0 to time dt
	
	solution = integrator->solution(); // extract the new gas properties
	
	toxhat(solution,f,nx,rusr); // normalize the gas properties
	
	if ( mode==2 ){
		myfnn(nx, x, fnn); // evaluate f^{MLP}
		for (int ii=0; ii<nx; ii++){f[ii] = f[ii] - x[ii] - fnn[ii];}}
		// f(x) is the increment of x minus f^{MLP}(x)
	else {for (int ii=0; ii<nx; ii++){f[ii] = f[ii] - x[ii];}}
	
	
	if (need[1]==1){ // this block is called when a Jacobian is needed
		
		double xp[nx]; double xm[nx];
		double fp[nf]; double fm[nf];
		
			
		for (int ii = 0; ii < nx; ii++){
			
			for( int jj = 0; jj < nx; jj++){xp[jj]=x[jj];xm[jj]=x[jj];}
			
			xp[ii] += dx; xm[ii] -= dx;
			
			tnow = 0.0;
			fromxhat(xp,ptcl,nx,rusr);
			T[0] = ptcl[0];
			for (int ii = 1; ii < nx; ii++){ Y[ii-1] = ptcl[ii];}
			gas->setState_TPY(T[0], p, Y);   
			integrator->initialize(tnow, odes);
			integrator->integrate(dt);
			solution = integrator->solution();
			toxhat(solution,fp,nx,rusr);
			if (mode==2){
				myfnn(nx, xp, fnn);
				for (int ii=0; ii<nx; ii++){fp[ii] = fp[ii] - xp[ii] - fnn[ii];}}
			// evaluate f(x) at x+dx in the jj+1-th entry of x
			else {for (int ii=0; ii<nx; ii++){fp[ii] = fp[ii] - xp[ii];}}
			
			tnow = 0.0;
			fromxhat(xm,ptcl,nx,rusr);
			T[0] = ptcl[0];
			for (int ii = 1; ii < nx; ii++){ Y[ii-1] = ptcl[ii];}
			gas->setState_TPY(T[0], p, Y);   
			integrator->initialize(tnow, odes);
			integrator->integrate(dt);
			solution = integrator->solution();
			toxhat(solution,fm,nx,rusr);
			if (mode == 2){
				myfnn(nx, xm, fnn);
				for (int ii=0; ii<nx; ii++){fm[ii] = fm[ii] - xm[ii] - fnn[ii];}}
				// evaluate f(x) at x-dx in the jj+1-th entry of x
			else {for (int ii=0; ii<nx; ii++){fm[ii] = fm[ii] - xm[ii];}}
			
			for (int jj =0; jj< nx; jj++){g[jj+ii*(nx)] = 1.0*(fp[jj] - fm[jj])/(2*dx);}
			// calculate the jj+1-th partial derivative
			
		}
		
	}	
		
}

void mymix(int &nx, double ptcl1[], double ptcl2[], double alpha[], int iusr[], double rusr[] ){
	// mix two particles, conserving mass and energy
	
	double Y1[nx-1],Y2[nx-1]; // mass fractions
	double H1, H2; // enthalpies
	double T1[1], T2[1]; // temperatures
	double d; // work variable
	double p = OneAtm; //rusr[2*nx+4];
	
	
	T1[0] = ptcl1[0]; for ( int ii = 1; ii < nx; ii++ ){Y1[ii-1]=ptcl1[ii];}
	T2[0] = ptcl2[0]; for ( int ii = 1; ii < nx; ii++ ){Y2[ii-1]=ptcl2[ii];}
	// extract temperature and mass fractios for the two particles
	
	gas->setState_TPY(T1[0], p, Y1); // initialize the gas to the state of the first particle   
	
	H1 = gas->enthalpy_mass(); // get the enthalpy of the first particle
	
	gas->setState_TPY(T2[0], p, Y2); // initialize the gas to the state of the second particle
	
	H2 = gas->enthalpy_mass(); // get the enthalpy of the second particle
	
	d = H2 - H1; H1 += alpha[0]*d; H2 -= alpha[0]*d; //mix enthalpies
	// alpha is amount by which to mix the particles (0 is no mixing, 0.5 is complete mixing)
	
	for (int ii=0; ii<nx-1; ii++){
		d = Y2[ii] - Y1[ii]; Y1[ii] += alpha[0]*d; Y2[ii] -= alpha[0]*d; // mix mass fractions		
	}
	
	d = alpha[0]*(T2-T1);
	
	gas->setState_TPY(T1[0] + d, p, Y1);
	gas->setState_HP(H1, p);
	
	T1[0]=gas->temperature();
	
	gas->setState_TPY(T2[0] - d, p, Y2);
	gas->setState_HP(H2, p);
	
	T2[0]=gas->temperature(); // set the particle's thermodynamic states to the new mixed values, and 
	// extract the corresponding temperatures
	
	ptcl1[0] = T1[0]; for ( int ii = 1; ii < nx; ii++ ){ptcl1[ii]=Y1[ii-1];}
	ptcl2[0] = T2[0]; for ( int ii = 1; ii < nx; ii++ ){ptcl2[ii]=Y2[ii-1];} // pass the new temperatures and mass fractions to the output
	
	
	
	
}
