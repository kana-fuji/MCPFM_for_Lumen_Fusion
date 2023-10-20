/////////////////////////////////////////////////////////////////////////////////
//
//  multicellular phase field model in 2D using cuda
//
/////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include <iomanip> //setprecision 
#include <time.h>
#include "SFMT.h"
//#include <curand.h>
//#include <curand_kernel.h>

#include <omp.h>

#include "config.cuh"
using namespace std;

const int WARP = 32; //number of threads in a wArp

const int IMAX = 2048;
const int JMAX = 2048;

const int CIMAX = 512;
const int CJMAX = 512;

const float DX = 0.01f;
const float DY = 0.01f;

const float T     =  900000.0f;
const float TOUT  =     100.0f;
const float TOUT2 =     100.0f;
const float DT    =       0.02f;
const float DTS   =       1.0f;

const int NMAX = 64;

PARAM::psys  h_sys;
PARAM::param h_para;
PARAM::cells h_cells[NMAX];
PARAM::dtv   h_dtv[NMAX];
PARAM::com   h_ijk[NMAX];
float h_xi;

__constant__ PARAM::psys  d_sys;
__constant__ PARAM::param d_para;
__constant__ PARAM::cells d_cells[NMAX];
__constant__ PARAM::com   d_ijk[NMAX];
__constant__ float d_xi;
__constant__ int   d_cnum;


//Mersenne twister---------------------------
sfmt_t sfmt;
void my_srand( uint32_t seed ) {
    sfmt_init_gen_rand( &sfmt, seed );
}
float MT_rand(){
  return (float)sfmt_genrand_res53(&sfmt);
}
//Mersenne twister---------------------------end


//Quick sort---------------------------
typedef float value_type; //type of key for sort

//return median of x, y, z
value_type med3(value_type x, value_type y, value_type z) {
  if (x < y) {
    if (y < z) return y; else if (z < x) return x; else return z;
  } else {
    if (z < y) return y; else if (x < z) return x; else return z;
  }
}

//quicksort
// a     : array
// left  : start position of array
// right : end position of array
void quicksort(value_type a[], int left, int right) {
  if (left < right) {
    int i = left, j = right;
    value_type tmp, pivot = med3(a[i], a[i + (j - i) / 2], a[j]); // overfrow on (i+j)/2
    while (1) { //devide a[] for clasters pivot over and less
      while (a[i] < pivot) i++; //surch a[i] >= pivot 
      while (pivot < a[j]) j--; //surch a[j] <= pivot
      if (i >= j) break;
      tmp = a[i]; a[i] = a[j]; a[j] = tmp; // change a[i] and a[j] 
      i++; j--;
    }
    quicksort(a, left, i - 1);  //sort left
    quicksort(a, j + 1, right); //sort right
  }
}
//Quick sort---------------------------end

////////////////////////////
//
//  Kernel function
//
////////////////////////////

__global__
void init_u_Kernel(float* u,float* r0){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % d_sys.cimax;
  int y = (index / d_sys.cimax) % d_sys.cjmax;
  int n = ((index / d_sys.cimax) / d_sys.cjmax) % d_cnum;


  float center_x=d_sys.cjmax*0.5f*d_sys.dx;
  float center_y=d_sys.cjmax*0.5f*d_sys.dy;
  float r;

  float xx,yy;

  xx=__fmul_rn(__fsub_rn(__fmul_rn(__fadd_rn(__int2float_rn(x),0.5f),d_sys.dx),center_x),
	       __fsub_rn(__fmul_rn(__fadd_rn(__int2float_rn(x),0.5f),d_sys.dx),center_x));
  yy=__fmul_rn(__fsub_rn(__fmul_rn(__fadd_rn(__int2float_rn(y),0.5f),d_sys.dx),center_y),
	       __fsub_rn(__fmul_rn(__fadd_rn(__int2float_rn(y),0.5f),d_sys.dx),center_y));

  r=__fsqrt_rn(__fadd_rn(xx,yy));
  u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]=__fmul_rn(__fsub_rn(1.0f,
								   tanhf(__fdiv_rn(__fsub_rn(r,r0[n]),
										   __fsqrt_rn(2.0f*d_para.D_u)))),
							 0.5f);
}



__device__
float h(const float pf){
  return __fmul_rn(__fmul_rn(pf,pf),(__fsub_rn(3.0f,2.0f*pf)));
}



__device__
float laplacian4(const float v1,
		 const float vx0,const float vx2,
		 const float vy0,const float vy2
		 ){
  return __fdiv_rn(__fadd_rn(__fsub_rn(__fadd_rn(vx2,vx0),__fadd_rn(v1,v1)),
			     __fsub_rn(__fadd_rn(vy2,vy0),__fadd_rn(v1,v1))),
		   __fmul_rn(d_sys.dx,d_sys.dx));
}


__device__
float laplacian8(const float v1,
		 const float vx0,const float vx2,const float vy0,const float vy2,
		 const float vx0y0,const float vx2y0,const float vx0y2,const float vx2y2
		 ){

  // return __fmul_rn(__fadd_rn(
  //  			     __fadd_rn(
  //  				       __fdiv_rn(__fsub_rn(__fadd_rn(vx2,vx0),__fadd_rn(v1,v1)),
  //  						 __fmul_rn(d_sys.dx,d_sys.dx)),
  //  				       __fdiv_rn(__fsub_rn(__fadd_rn(vy2,vy0),__fadd_rn(v1,v1)),
  //  						 __fmul_rn(d_sys.dx,d_sys.dx))
  //  				       ),
  //  			     __fdiv_rn(__fsub_rn(__fmul_rn(__fadd_rn(__fadd_rn(vx2y0,vx0y0),
  //  								     __fadd_rn(vx0y2,vx2y2)),
  //  							   0.5f),
  //  						 __fmul_rn(2.0f,v1)),
  //  				       __fmul_rn(d_sys.dx,d_sys.dx))
  //  			     ),
  //  		   0.5f);
  return
    ((vx0-2.0f*v1+vx2)/(d_sys.dx*d_sys.dx)
     +(vy0-2.0f*v1+vy2)/(d_sys.dy*d_sys.dy)
     +(0.5f*(vx0y0+vx2y0+vx0y2+vx2y2)-2.0f*v1)/(d_sys.dy*d_sys.dy)
     )*0.5f;

}






__global__
void time_evolution_Kernel(float** pf_temp,float** pf,float *phi,int row){

  int n = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cjmax;
  int x = (index / d_sys.cjmax) % d_sys.cimax;


  if(x>0 && x<d_sys.cimax-1 && y>0 && y<d_sys.cimax-1){
    pf_temp[n][x+d_sys.cimax*y]
      =pf[n][x+d_sys.cimax*y]
      +(
	//d_para.D_u*laplacian8(x,y,row,pf[n])
	d_para.D_u*(
		    (pf[n][x+row*(y-1)]-2.00f*pf[n][x+row*y]+pf[n][x+row*(y+1)])
		    /(d_sys.dx*d_sys.dx)
		    +(pf[n][(x-1)+row*y]-2.00f*pf[n][x+row*y]+pf[n][(x+1)+row*y])
		    /(d_sys.dy*d_sys.dy)
		    +(0.50f*
		      (pf[n][(x-1)+row*(y-1)]+pf[n][(x-1)+row*(y+1)]
		       +pf[n][(x+1)+row*(y-1)]+pf[n][(x+1)+row*(y+1)])-2.0f*pf[n][x+row*y])
		    /(d_sys.dx*d_sys.dy)
		    )*0.50f
	+pf[n][x+d_sys.cimax*y]*(1-pf[n][x+d_sys.cimax*y])
	*(
	  pf[n][x+d_sys.cimax*y]-0.50f
	  //+d_para.alpha*(d_para.V-d_cells[n].v)
	  +d_para.alpha*(d_cells[n].targetv-d_cells[n].v)
	  -d_para.beta*(phi[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]-h(pf[n][x+d_sys.cimax*y]))
	  +d_para.eta*(
		       (
			(phi[(x+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]
			 -h(pf[n][x+d_sys.cimax*(y-1)]))
			-2.00f*(phi[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
				-h(pf[n][x+d_sys.cimax*y]))
			+(phi[(x+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]
			 -h(pf[n][x+d_sys.cimax*(y+1)]))
			)
		       /(d_sys.dx*d_sys.dx)
		       +(
			 (phi[((x-1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
			  -h(pf[n][(x-1)+d_sys.cimax*y]))
			-2.00f*(phi[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
				-h(pf[n][x+d_sys.cimax*y]))
			 +(phi[((x+1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
			   -h(pf[n][(x+1)+d_sys.cimax*y]))
			 )
		       /(d_sys.dy*d_sys.dy)
		       +(0.50f*
			 (
			  (phi[((x-1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]
			   -h(pf[n][(x-1)+d_sys.cimax*(y-1)]))
			  +(phi[((x-1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]
			   -h(pf[n][(x-1)+d_sys.cimax*(y+1)]))
			  +(phi[((x+1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]
			   -h(pf[n][(x+1)+d_sys.cimax*(y-1)]))
			  +(phi[((x+1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]
			   -h(pf[n][(x+1)+d_sys.cimax*(y+1)]))
			  )
			 -2.0f*(phi[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
				-h(pf[n][x+d_sys.cimax*y]))
			 )
		       /(d_sys.dx*d_sys.dy)
		       )*0.50f
	  +d_para.gamma*(
			 (pf[n][x+row*(y-1)]-2.00f*pf[n][x+row*y]+pf[n][x+row*(y+1)])
			 /(d_sys.dx*d_sys.dx)
			 +(pf[n][(x-1)+row*y]-2.00f*pf[n][x+row*y]+pf[n][(x+1)+row*y])
			 /(d_sys.dy*d_sys.dy)
			 +(0.50f*
			   (pf[n][(x-1)+row*(y-1)]+pf[n][(x-1)+row*(y+1)]
			    +pf[n][(x+1)+row*(y-1)]+pf[n][(x+1)+row*(y+1)])-2.0f*pf[n][x+row*y])
			 /(d_sys.dx*d_sys.dy)
			 )*0.50f
	  )
	)*d_sys.dt/d_para.tau_u
      ;
  }
}


__global__
void time_evolution_with_reshaping_Kernel(float** pf_temp,float** pf,float *phi,float *c,int row){

  int n = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cjmax;
  int x = (index / d_sys.cjmax) % d_sys.cimax;


  if(x>0 && x<d_sys.cimax-1 && y>0 && y<d_sys.cimax-1){
    pf_temp[n][x+d_sys.cimax*y]
      =pf[n][x+d_sys.cimax*y]
      +(
	pf[n][x+d_sys.cimax*y]*(1-pf[n][x+d_sys.cimax*y])
	*(
	  //d_para.alpha*(d_para.V-d_cells[n].v)
	  +d_para.alpha*(d_cells[n].targetv-d_cells[n].v)
	  -d_para.beta*(phi[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]-h(pf[n][x+d_sys.cimax*y]))
	  +d_para.eta*(
		       (
			(phi[(x+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]
			 -h(pf[n][x+d_sys.cimax*(y-1)]))
			-2.00f*(phi[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
				-h(pf[n][x+d_sys.cimax*y]))
			+(phi[(x+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]
			 -h(pf[n][x+d_sys.cimax*(y+1)]))
			)
		       /(d_sys.dx*d_sys.dx)
		       +(
			 (phi[((x-1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
			  -h(pf[n][(x-1)+d_sys.cimax*y]))
			-2.00f*(phi[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
				-h(pf[n][x+d_sys.cimax*y]))
			 +(phi[((x+1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
			   -h(pf[n][(x+1)+d_sys.cimax*y]))
			 )
		       /(d_sys.dy*d_sys.dy)
		       +(0.50f*
			 (
			  (phi[((x-1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]
			   -h(pf[n][(x-1)+d_sys.cimax*(y-1)]))
			  +(phi[((x-1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]
			   -h(pf[n][(x-1)+d_sys.cimax*(y+1)]))
			  +(phi[((x+1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]
			   -h(pf[n][(x+1)+d_sys.cimax*(y-1)]))
			  +(phi[((x+1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]
			   -h(pf[n][(x+1)+d_sys.cimax*(y+1)]))
			  )
			 -2.0f*(phi[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
				-h(pf[n][x+d_sys.cimax*y]))
			 )
		       /(d_sys.dx*d_sys.dy)
		       )*0.50f
	  )
	)*d_sys.dt/d_para.tau_u
      +2.0f*d_para.gamma_curv*sqrtf(d_para.D_u)
      *(
	(pf[n][x+row*(y-1)]-2.00f*pf[n][x+row*y]+pf[n][x+row*(y+1)])
	/(d_sys.dx*d_sys.dx)
	+(pf[n][(x-1)+row*y]-2.00f*pf[n][x+row*y]+pf[n][(x+1)+row*y])
	/(d_sys.dy*d_sys.dy)
	+(0.50f*
	  (pf[n][(x-1)+row*(y-1)]+pf[n][(x-1)+row*(y+1)]
	   +pf[n][(x+1)+row*(y-1)]+pf[n][(x+1)+row*(y+1)])
	  -2.0f*pf[n][x+row*y]
	  )/(d_sys.dx*d_sys.dy)
	)*0.50f
      *d_sys.dt/d_para.tau_u
      ;

    //interaction with medium
    pf_temp[n][x+d_sys.cimax*y]
      +=pf[n][x+d_sys.cimax*y]*(1-pf[n][x+d_sys.cimax*y])
      *(
	-d_para.beta_cu*c[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
	+d_para.eta_cu*(
			(c[(x+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]
			 -2.00f*c[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
			 +c[(x+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]
			 )/(d_sys.dx*d_sys.dx)
		       +(c[((x-1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
			 -2.00f*c[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
			 +c[((x+1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
			 )/(d_sys.dy*d_sys.dy)
		       +(0.50f*
			 (
			  c[((x-1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]
			  +c[((x-1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]
			  +c[((x+1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]
			  +c[((x+1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]
			  )
			 -2.0f*c[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
			 )/(d_sys.dx*d_sys.dy)
			)*0.50f
	)*d_sys.dt/d_para.tau_u
      ;
  }
}


__global__
void time_evolution_u_with_reshaping_Kernel(float* u_temp,float* u,float *phi,float *s,float *c){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % d_sys.cimax;
  int y = (index / d_sys.cimax) % d_sys.cjmax;
  int n = ((index / d_sys.cimax) / d_sys.cjmax) % d_cnum;

  float dttau=__fdiv_rn(d_sys.dt,d_para.tau_u);

  if(x>0 && x<d_sys.cimax-1 && y>0 && y<d_sys.cimax-1){
    //cells
    u_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
      =u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
      +(
	 //diffusion
	 // d_para.D_u*laplacian8(u[n][x+row*y],
	 // 			  u[n][(x-1)+row*y],u[n][(x+1)+row*y],
	 // 			  u[n][x+row*(y-1)],u[n][x+row*(y+1)],
	 // 			  u[n][(x-1)+row*(y-1)],u[n][(x+1)+row*(y-1)],
	 // 			  u[n][(x-1)+row*(y+1)],u[n][(x+1)+row*(y+1)])

	 u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
	 *(1.0f-u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax])
	 *(
	   //double well potential
	   //(u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]-0.50f)

	   //volume conservation
	   //+d_para.alpha*(d_para.V-d_cells[n].v)
	   d_para.alpha*(d_cells[n].targetv-d_cells[n].v)


	   //excluded volume
	   -1.0f*d_para.beta
	   *(phi[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
	     -h(u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax])
	     )
			
	   //cell-cell adhesion
	   +d_para.eta*
	   laplacian8(phi[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
		      -h(u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]),
		      phi[((x-1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
		      -h(u[(x-1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]),
		      phi[((x+1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
		      -h(u[(x+1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]),
		      phi[(x+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]
		      -h(u[x+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax]),
		      phi[(x+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]
		      -h(u[x+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax]),
		      phi[((x-1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]
		      -h(u[(x-1)+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax]),
		      phi[((x+1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]
		      -h(u[(x+1)+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax]),
		      phi[((x-1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]
		      -h(u[(x-1)+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax]),
		      phi[((x+1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]
		      -h(u[(x+1)+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax]))

	   //curveture effect
	   +12.0f*d_para.gamma_curv*__fsqrt_rn(d_para.D_u)*
	   laplacian8(h(u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]),
	    	      h(u[(x-1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]),
	    	      h(u[(x+1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]),
	    	      h(u[x+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax]),
	    	      h(u[x+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax]),
	    	      h(u[(x-1)+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax]),
	    	      h(u[(x+1)+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax]),
	    	      h(u[(x-1)+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax]),
	    	      h(u[(x+1)+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax]))

	   )
	 )*dttau;

    //interaction with lumen
    u_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
      +=u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
      *(1.0f-u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax])
      *(
    	-d_para.beta_s*h(s[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)])
    	+d_para.eta_s*laplacian8(h(s[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]),
    				 h(s[((x-1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]),
    				 h(s[((x+1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]),
    				 h(s[(x+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]),
    				 h(s[(x+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]),
    				 h(s[((x-1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]),
    				 h(s[((x+1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]),
    				 h(s[((x-1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]),
    				 h(s[((x+1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]))
    	)*dttau;

    //interaction with medium
    u_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
      +=u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
      *(1.0f-u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax])
      *(
    	-d_para.beta_cu*h(c[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)])
    	+d_para.eta_cu*laplacian8(h(c[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]),
				  h(c[((x-1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]),
				  h(c[((x+1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]),
				  h(c[(x+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]),
				  h(c[(x+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]),
				  h(c[((x-1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]),
				  h(c[((x+1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]),
				  h(c[((x-1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]),
				  h(c[((x+1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]))
    	)*dttau;

  }
}



__global__
void time_evolution_up_with_reshaping_Kernel_test(float *u_temp,float *u,
						  float *p_temp,float *p,
						  float *phi,float*pall,float *s,float *c,
						  float *u_adhe){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % d_sys.cimax;
  int y = (index / d_sys.cimax) % d_sys.cjmax;
  int n = ((index / d_sys.cimax) / d_sys.cjmax) % d_cnum;

  float dttau=__fdiv_rn(d_sys.dt,d_para.tau_u);

  if(x>0 && x<d_sys.cimax-1 && y>0 && y<d_sys.cimax-1){
    //cells
    u_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
      =u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
      +(
	 //diffusion
	 // d_para.D_u*laplacian8(u[n][x+row*y],
	 // 			  u[n][(x-1)+row*y],u[n][(x+1)+row*y],
	 // 			  u[n][x+row*(y-1)],u[n][x+row*(y+1)],
	 // 			  u[n][(x-1)+row*(y-1)],u[n][(x+1)+row*(y-1)],
	 // 			  u[n][(x-1)+row*(y+1)],u[n][(x+1)+row*(y+1)])

	 u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
	 *(1.0f-u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax])
	 *(
	   //double well potential
	   //(u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]-0.50f)

	   //volume conservation
	   //+d_para.alpha*(d_para.V-d_cells[n].v)
	   d_para.alpha*(d_cells[n].targetv-d_cells[n].v)


	   //excluded volume
	   -1.0f*d_para.beta
	   *(phi[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
	     -h(u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax])
	     )
			

	   //cell-cell adhesion with anti-adhesion term 1
	   +d_para.eta/2.0f*(1+tanhf((d_para.p_th
				     -pall[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)])
				    /d_para.l_anti))
	   *laplacian8(phi[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
		       -h(u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]),
		       phi[((x-1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
		       -h(u[(x-1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]),
		       phi[((x+1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
		       -h(u[(x+1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]),
		       phi[(x+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]
		       -h(u[x+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax]),
		       phi[(x+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]
		       -h(u[x+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax]),
		       phi[((x-1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]
		       -h(u[(x-1)+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax]),
		       phi[((x+1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]
		       -h(u[(x+1)+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax]),
		       phi[((x-1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]
		       -h(u[(x-1)+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax]),
		       phi[((x+1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]
		       -h(u[(x+1)+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax]))


	   //cell-cell adhesion with anti-adhesion term 2
	    +(
	      (pall[((x+1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]-
	       pall[((x-1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)])
	      /(2.0f*d_sys.dx)
	      *((phi[((x+1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
	    	-h(u[(x+1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]))-
	        (phi[((x-1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]
	    	-h(u[(x-1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax])))
	      /(2.0f*d_sys.dx)
	      +
	      (pall[(x+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]-
	       pall[(x+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)])
	      /(2.0f*d_sys.dy)
	      *((phi[(x+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]
	    	-h(u[x+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax]))-
	        (phi[(x+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]
	    	-h(u[x+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax])))
	      /(2.0f*d_sys.dy)
	      )
	    *d_para.eta/(4.0f*d_para.l_anti
	    		*cosh((d_para.p_th-pall[(x+d_cells[n].cimin)
	    					+d_sys.imax*(y+d_cells[n].cjmin)])/d_para.l_anti)
	    		*cosh((d_para.p_th-pall[(x+d_cells[n].cimin)
	    					+d_sys.imax*(y+d_cells[n].cjmin)])/d_para.l_anti))


	   //curveture effect
	   +12.0f*d_para.gamma_curv*__fsqrt_rn(d_para.D_u)*
	   laplacian8(u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax],
	    	      u[(x-1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax],
	    	      u[(x+1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax],
	    	      u[x+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax],
	    	      u[x+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax],
	    	      u[(x-1)+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax],
	    	      u[(x+1)+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax],
	    	      u[(x-1)+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax],
	    	      u[(x+1)+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax])

	   )
	 )*dttau;

    //interaction with lumen
    u_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
      +=u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
      *(1.0f-u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax])
      *(
    	-d_para.beta_s*h(s[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)])
    	+d_para.eta_s*laplacian8(h(s[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]),
    				 h(s[((x-1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]),
    				 h(s[((x+1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]),
    				 h(s[(x+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]),
    				 h(s[(x+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]),
    				 h(s[((x-1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]),
    				 h(s[((x+1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]),
    				 h(s[((x-1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]),
    				 h(s[((x+1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]))
    	)*dttau;

    //interaction with medium
    u_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
      +=u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
      *(1.0f-u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax])
      *(
    	-d_para.beta_cu*h(c[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)])
    	+d_para.eta_cu*laplacian8(h(c[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]),
				  h(c[((x-1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]),
				  h(c[((x+1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]),
				  h(c[(x+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]),
				  h(c[(x+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]),
				  h(c[((x-1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]),
				  h(c[((x+1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]),
				  h(c[((x-1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]),
				  h(c[((x+1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]))
    	)*dttau;



    //anti-adhesive molecules
    p_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
      =p[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
      +(
	//diffusion
	d_para.D_p*laplacian8(p[x+y*d_sys.cimax+n*d_sys.cimax*d_sys.cjmax],
			      p[(x-1)+y*d_sys.cimax+n*d_sys.cimax*d_sys.cjmax],
			      p[(x+1)+y*d_sys.cimax+n*d_sys.cimax*d_sys.cjmax],
			      p[x+(y-1)*d_sys.cimax+n*d_sys.cimax*d_sys.cjmax],
			      p[x+(y+1)*d_sys.cimax+n*d_sys.cimax*d_sys.cjmax],
			      p[(x-1)+(y-1)*d_sys.cimax+n*d_sys.cimax*d_sys.cjmax],
			      p[(x+1)+(y-1)*d_sys.cimax+n*d_sys.cimax*d_sys.cjmax],
			      p[(x-1)+(y+1)*d_sys.cimax+n*d_sys.cimax*d_sys.cjmax],
			      p[(x+1)+(y+1)*d_sys.cimax+n*d_sys.cimax*d_sys.cjmax])

	//interface
	+p[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
	*(1.0f-p[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax])
	*(
	  //double well potential
	  d_para.k_p*(p[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]-0.50f)

	  // //curveture effect
	  // +d_para.gamma_p
	  // *laplacian8(p[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax],
	  //   	      p[(x-1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax],
	  //   	      p[(x+1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax],
	  //   	      p[x+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax],
	  //   	      p[x+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax],
	  //   	      p[(x-1)+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax],
	  //   	      p[(x+1)+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax],
	  //   	      p[(x-1)+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax],
	  //   	      p[(x+1)+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax])
	  )

	//cell-cell adhesion with anti-adhesion
	+u_adhe[(x+d_cells[n].cimin)]//d_para.eta/6.0f*\nabla h(u_m).\nabla h(u_n)
	/(4.0f*d_para.l_anti
	  *cosh((d_para.p_th-pall[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)])
		/d_para.l_anti)
	  *cosh((d_para.p_th-pall[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)])
		/d_para.l_anti))

	//volume conservation
	+d_para.alpha_p*(d_para.Vp-d_cells[n].vp)

	//localization on cell membrane
	+d_para.C_p*u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
	*(1.0f-u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax])

        //apical-lumen adhesion
	+d_para.eta_ps/6.0f
	*laplacian8(h(s[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]),
		    h(s[((x-1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]),
		    h(s[((x+1)+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]),
		    h(s[(x+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]),
		    h(s[(x+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]),
		    h(s[((x-1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]),
		    h(s[((x+1)+d_cells[n].cimin)+d_sys.imax*((y-1)+d_cells[n].cjmin)]),
		    h(s[((x-1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]),
		    h(s[((x+1)+d_cells[n].cimin)+d_sys.imax*((y+1)+d_cells[n].cjmin)]))

	)*d_sys.dt/d_para.tau_p
      ;
  }
}





__global__
void time_evolution_c_with_reshaping_Kernel(float* pf_temp,float* pf,float *phi,int row){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.jmax;
  int x = (index / d_sys.jmax) % d_sys.imax;


  if(x>0 && x<d_sys.imax-1 && y>0 && y<d_sys.imax-1){
    pf_temp[x+d_sys.imax*y]
      =pf[x+d_sys.imax*y]
      +(
	pf[x+d_sys.imax*y]*(1-pf[x+d_sys.imax*y])
	*(
	  d_para.xi_c
	  -d_para.beta_cu*phi[x+d_sys.imax*y]
	  +d_para.eta_cu*(
			  (phi[x+d_sys.imax*(y-1)]-2.00f*phi[x+d_sys.imax*y]+phi[x+d_sys.imax*(y+1)]
			   )/(d_sys.dx*d_sys.dx)
			  +(phi[(x-1)+d_sys.imax*y]-2.00f*phi[x+d_sys.imax*y]+phi[(x+1)+d_sys.imax*y]
			    )/(d_sys.dy*d_sys.dy)
			  +(0.50f*
			    (phi[(x-1)+d_sys.imax*(y-1)]+phi[(x-1)+d_sys.imax*(y+1)]
			     +phi[(x+1)+d_sys.imax*(y-1)]+phi[(x+1)+d_sys.imax*(y+1)])
			    -2.0f*phi[x+d_sys.imax*y]
			    )/(d_sys.dx*d_sys.dy)
			  )*0.50f
	  )
	)*d_sys.dt/d_para.tau_c
      ;
  }
}



__global__
void time_evolution_sc_with_reshaping_Kernel(float* c_temp,float* c,float* s_temp,float* s,
					     float *phi,float* uall,float* pall){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.jmax;
  int x = (index / d_sys.jmax) % d_sys.imax;


  if(x>0 && x<d_sys.imax-1 && y>0 && y<d_sys.jmax-1){
    //medium
    c_temp[x+d_sys.imax*y]
      =c[x+d_sys.imax*y]
      +(
	c[x+d_sys.imax*y]*(1.0f-c[x+d_sys.imax*y])
	*(
	  //pressuer
	  d_para.xi_c

	  //excluded volume with cells
	  -d_para.beta_cu*phi[x+d_sys.imax*y]

	  //excluded volume with lumen
	  -d_para.beta_cs*h(s[x+d_sys.imax*y])

	  //cell-medium adhesion
	  +d_para.eta_cu*laplacian8(phi[x+d_sys.imax*y],
				    phi[(x-1)+d_sys.imax*y],phi[(x+1)+d_sys.imax*y],
				    phi[x+d_sys.imax*(y-1)],phi[x+d_sys.imax*(y+1)],
				    phi[(x-1)+d_sys.imax*(y-1)],phi[(x+1)+d_sys.imax*(y-1)],
				    phi[(x-1)+d_sys.imax*(y+1)],phi[(x+1)+d_sys.imax*(y+1)])


	  )
	)*d_sys.dt/d_para.tau_c
      ;

    //lumen
    s_temp[x+d_sys.imax*y]
      =s[x+d_sys.imax*y]
      +(
	s[x+d_sys.imax*y]*(1.0f-s[x+d_sys.imax*y])
	*(
	  //pressuer
	  d_para.xi
	  //d_xi

	  //excluded volume with cells
	  -d_para.beta_s*phi[x+d_sys.imax*y]

	  //excluded volume with medium
	  -d_para.beta_cs*h(c[x+d_sys.imax*y])

	  //cell-lumen adhesion
	  +d_para.eta_s*laplacian8(phi[x+d_sys.imax*y],
				   phi[(x-1)+d_sys.imax*y],phi[(x+1)+d_sys.imax*y],
				   phi[x+d_sys.imax*(y-1)],phi[x+d_sys.imax*(y+1)],
				   phi[(x-1)+d_sys.imax*(y-1)],phi[(x+1)+d_sys.imax*(y-1)],
				   phi[(x-1)+d_sys.imax*(y+1)],phi[(x+1)+d_sys.imax*(y+1)])

	  )
	)*d_sys.dt/d_para.tau_s
      ;

    //local volume conservation
    if(pall[x+d_sys.imax*y]>d_para.p_st &&
       s[x+d_sys.imax*y]+uall[x+d_sys.imax*y]+c[x+d_sys.imax*y]<d_para.v_t){
      s_temp[x+d_sys.imax*y]
	+=d_para.alpha_s*(1.0f-(s[x+d_sys.imax*y]+uall[x+d_sys.imax*y]+c[x+d_sys.imax*y]))
				*d_sys.dt/d_para.tau_s;
    }
  }
}


__global__
void time_evolution_sc_with_reshaping_2_Kernel(float* c_temp,float* c,float* s_temp,float* s,
					       float *phi,float* uall,
					       int imin,int jmin,int imax,int jmax,
					       float vc){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % imax;
  int y = (index / imax) % jmax;


  if(x+imin>0 && x+imin<d_sys.imax-1 && y+jmin>0 && y+jmin<d_sys.jmax-1){
    //medium
    c_temp[(x+imin)+d_sys.imax*(y+jmin)]
      =c[(x+imin)+d_sys.imax*(y+jmin)]
      +(
	c[(x+imin)+d_sys.imax*(y+jmin)]*(1.0f-c[(x+imin)+d_sys.imax*(y+jmin)])
	*(
	  //pressuer
	  d_para.xi_c

	  //excluded volume with cells
	  -d_para.beta_cu*phi[(x+imin)+d_sys.imax*(y+jmin)]

	  //excluded volume with lumen
	  -d_para.beta_cs*h(s[(x+imin)+d_sys.imax*(y+jmin)])

	  //cell-medium adhesion
	  +d_para.eta_cu*laplacian8(phi[(x+imin)+d_sys.imax*(y+jmin)],
				    phi[(x+imin-1)+d_sys.imax*(y+jmin)],phi[(x+imin+1)+d_sys.imax*(y+jmin)],
				    phi[(x+imin)+d_sys.imax*(y+jmin-1)],phi[(x+imin)+d_sys.imax*(y+jmin+1)],
				    phi[(x+imin-1)+d_sys.imax*(y+jmin-1)],phi[(x+imin+1)+d_sys.imax*(y+jmin-1)],
				    phi[(x+imin-1)+d_sys.imax*(y+jmin+1)],phi[(x+imin+1)+d_sys.imax*(y+jmin+1)])

          //20220520 Bulk modulus
          +d_para.alpha_c*(d_sys.imax*d_sys.jmax*d_sys.dx*d_sys.dy-vc)

	  )
	)*d_sys.dt/d_para.tau_c
      ;

    //lumen
    s_temp[(x+imin)+d_sys.imax*(y+jmin)]
      =s[(x+imin)+d_sys.imax*(y+jmin)]
      +(
	s[(x+imin)+d_sys.imax*(y+jmin)]*(1.0f-s[(x+imin)+d_sys.imax*(y+jmin)])
	*(
	  //pressuer
	  d_para.xi
	  //d_xi

	  //excluded volume with cells
	  -d_para.beta_s*phi[(x+imin)+d_sys.imax*(y+jmin)]

	  //excluded volume with medium
	  -d_para.beta_cs*h(c[(x+imin)+d_sys.imax*(y+jmin)])

	  //cell-lumen adhesion
	  +d_para.eta_s*laplacian8(phi[(x+imin)+d_sys.imax*(y+jmin)],
				   phi[(x+imin-1)+d_sys.imax*(y+jmin)],
				   phi[(x+imin+1)+d_sys.imax*(y+jmin)],
				   phi[(x+imin)+d_sys.imax*(y+jmin-1)],
				   phi[(x+imin)+d_sys.imax*(y+jmin+1)],
				   phi[(x+imin-1)+d_sys.imax*(y+jmin-1)],
				   phi[((x+imin)+1)+d_sys.imax*(y+jmin-1)],
				   phi[(x+imin-1)+d_sys.imax*(y+jmin+1)],
				   phi[((x+imin)+1)+d_sys.imax*(y+jmin+1)])

	  )
	)*d_sys.dt/d_para.tau_s
      ;

  }
}



__global__
void u_n_Kernel(float* u_n, float* u, int n){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  u_n[x+d_sys.cimax*y]=u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax];
}

__global__
void up_n_Kernel(float* u_n, float* u,float* p_n, float* p,int n){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  u_n[x+d_sys.cimax*y]=u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax];
  p_n[x+d_sys.cimax*y]=p[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax];
}


__global__
void update_Kernel(float* pf, float* pf_temp){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % d_sys.cimax;
  int y = (index / d_sys.cimax) % d_sys.cjmax;
  int n = ((index / d_sys.cimax) / d_sys.cjmax) % d_cnum;

  pf[x+y*d_sys.cimax+n*d_sys.cimax*d_sys.cjmax]=pf_temp[x+y*d_sys.cimax+n*d_sys.cimax*d_sys.cjmax];
}

__global__
void update_up_Kernel(float* u, float* u_temp,float* p, float* p_temp){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % d_sys.cimax;
  int y = (index / d_sys.cimax) % d_sys.cjmax;
  int n = ((index / d_sys.cimax) / d_sys.cjmax) % d_cnum;

  u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]=u_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax];
  p[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]=p_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax];
}


__global__
void update_all_Kernel(float* pf, float* pf_temp){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.imax;
  int x = (index / d_sys.imax) % d_sys.imax;

  pf[x+d_sys.imax*y]=pf_temp[x+d_sys.imax*y];
}

__global__
void update_all_2_Kernel(float* pf, float* pf_temp,int imin,int jmin,int imax,int jmax){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % imax;
  int y = (index / imax) % jmax;

  pf[(x+imin)+d_sys.imax*(y+jmin)]=pf_temp[(x+imin)+d_sys.imax*(y+jmin)];
}


__global__
void update_sc_Kernel(float* s, float* s_temp,float* c, float* c_temp){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.imax;
  int x = (index / d_sys.imax) % d_sys.imax;

  s[x+d_sys.imax*y]=s_temp[x+d_sys.imax*y];
  c[x+d_sys.imax*y]=c_temp[x+d_sys.imax*y];
}

__global__
void update_sc_2_Kernel(float* s, float* s_temp,float* c, float* c_temp,int imin,int jmin,int imax,int jmax){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % imax;
  int y = (index / imax) % jmax;

  s[(x+imin)+d_sys.imax*(y+jmin)]=s_temp[(x+imin)+d_sys.imax*(y+jmin)];
  c[(x+imin)+d_sys.imax*(y+jmin)]=c_temp[(x+imin)+d_sys.imax*(y+jmin)];
}


__global__
void CoM_Kernel(float** pf, float** pf_temp){
  int n = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  int ii,jj;
  ii=x+d_ijk[n].di; jj=y+d_ijk[n].dj;
  if(ii>0 && ii<d_sys.cimax-1 && jj>0 && jj<d_sys.cjmax-1)
    pf_temp[n][x+d_sys.cimax*y]=pf[n][ii+d_sys.cimax*jj];
  else
    pf_temp[n][x+d_sys.cimax*y]=0.0f;
}

__global__
void CoM_up_Kernel(float* u, float* u_temp,float* p, float* p_temp){
  int n = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  int ii,jj;
  ii=x+d_ijk[n].di; jj=y+d_ijk[n].dj;
  if(ii>0 && ii<d_sys.cimax-1 && jj>0 && jj<d_sys.cjmax-1){
    u_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
      =u[ii+d_sys.cimax*jj+n*d_sys.cimax*d_sys.cjmax];
    p_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
      =p[ii+d_sys.cimax*jj+n*d_sys.cimax*d_sys.cjmax];
  }
  else{
    u_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]=0.0f;
    p_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]=0.0f;
  }
}


__global__
void sumxy_Kernel(float* pf,float* pfx,float* pfy,int n){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  pfx[x+d_sys.cimax*y]=h(pf[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax])*(x+0.5f);
  pfy[x+d_sys.cimax*y]=h(pf[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax])*(y+0.5f);
}

__global__
void h_Kernel(float* pf_h,float* pf){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cimax;
  pf_h[x+d_sys.cimax*y]=h(pf[x+d_sys.cimax*y]);
}

__global__
void hu_Kernel(float* pf_h,float* pf){
  int n = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cimax;
  pf_h[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]=h(pf[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]);
}


__global__
void hv_Kernel(float* pf_h,float* pf){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cimax;
  pf_h[x+d_sys.cimax*y]=h(pf[x+d_sys.cimax*y])*d_sys.dx*d_sys.dy;
}


__global__
void reduce0(float *idata,float *odata, unsigned int n){

  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x; //1024
  unsigned int i = blockIdx.x * blockDim.x  + threadIdx.x; //(j*1024*2)+i(1024)

  sdata[tid] = (i < n) ? idata[i] : 0.0f;
  //sdata[tid] = (i < n) ? 0.1f : 0.0f;
  __syncthreads();
  //printf("%8.4f",sdata[tid]);

  for (unsigned int s=1; s<blockDim.x; s*=2) {
    if (tid % (2*s)==0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) odata[blockIdx.x]  = sdata[0];
}

__global__
void reduce3(float *idata,float *odata, unsigned int n){

  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x*2)  + threadIdx.x;

  float mysum = (i < n) ? idata[i] : 0.0f;
  if (i + blockDim.x < n) mysum += idata[i + blockDim.x];
  sdata[tid] = mysum;
  __syncthreads();

  for (unsigned int s=1; s<blockDim.x; s*=2) {
    if (tid % (2*s)==0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) odata[blockIdx.x]  = sdata[0];
}

__global__
void reduce3_i(int *idata,int *odata, unsigned int n){

  extern __shared__ int sdata_i[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x*2)  + threadIdx.x;

  int mysum = (i < n) ? idata[i] : 0;
  if (i + blockDim.x < n) mysum += idata[i + blockDim.x];
  sdata_i[tid] = mysum;
  __syncthreads();

  for (unsigned int s=1; s<blockDim.x; s*=2) {
    if (tid % (2*s)==0) {
      sdata_i[tid] += sdata_i[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) odata[blockIdx.x]  = sdata_i[0];
}




__global__
void init_all_Kernel(float* v){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.imax;
  int x = (index / d_sys.imax) % d_sys.jmax;

  v[x+d_sys.jmax*y]=0.0f;
}

__global__
void init_all_uphi_Kernel(float* u,float* phi){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.imax;
  int x = (index / d_sys.imax) % d_sys.jmax;

  u[x+d_sys.jmax*y]=0.0f;
  phi[x+d_sys.jmax*y]=0.0f;
}



__global__
void phi_Kernel(float* phi,const float* pf,const int n){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  phi[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]+=h(pf[x+d_sys.cjmax*y+n*d_sys.cimax*d_sys.cjmax]);
}

__global__
void u_m1_Kernel(float* u_n,float* hd_u_n,int n){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  u_n[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]=hd_u_n[x+d_sys.cimax*y];

}

__global__
void e_eta_Kernel(float* e_eta,float* pf,float* pf_n,int m){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  if(x>0 && x<d_sys.cimax-1 && y>0 && y<d_sys.cjmax-1){

    e_eta[(x+d_cells[m].cimin)+d_sys.jmax*(y+d_cells[m].cjmin)]
      +=(
	 (h(pf[(x+1)+d_sys.cimax*y+m*d_sys.cimax*d_sys.cjmax])
	  -h(pf[(x-1)+d_sys.cimax*y+m*d_sys.cimax*d_sys.cjmax]))*0.5f/d_sys.dx
	 *(h(pf_n[((x+1)+d_cells[m].cimin)+d_sys.jmax*(y+d_cells[m].cjmin)])
	   -h(pf_n[((x-1)+d_cells[m].cimin)+d_sys.jmax*(y+d_cells[m].cjmin)]))*0.5f/d_sys.dx
	 +(h(pf[x+d_sys.cimax*(y+1)+m*d_sys.cimax*d_sys.cjmax])
	   -h(pf[x+d_sys.cimax*(y-1)+m*d_sys.cimax*d_sys.cjmax]))*0.5f/d_sys.dy
	 *(h(pf_n[(x+d_cells[m].cimin)+d_sys.jmax*((y+1)+d_cells[m].cjmin)])
	   -h(pf_n[(x+d_cells[m].cimin)+d_sys.jmax*((y-1)+d_cells[m].cjmin)]))*0.5f/d_sys.dy
	 )*d_para.eta/6.0f
    ;

  }
}


__global__
void all_Kernel(float* uall,const float* pf,const int n){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  uall[(x+d_cells[n].cimin)+d_sys.jmax*(y+d_cells[n].cjmin)]+=pf[x+d_sys.cjmax*y+n*d_sys.cimax*d_sys.cjmax];
}


__global__
void all_uphi_Kernel(float* uall,float* phi,const float* u,int n){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % d_sys.cimax;
  int y = (index / d_sys.cimax) % d_sys.cjmax;

  uall[(x+d_cells[n].cimin)+d_sys.jmax*(y+d_cells[n].cjmin)]
    +=u[x+d_sys.cjmax*y+n*d_sys.cimax*d_sys.cjmax];
  phi[(x+d_cells[n].cimin)+d_sys.jmax*(y+d_cells[n].cjmin)]
    +=h(u[x+d_sys.cjmax*y+n*d_sys.cimax*d_sys.cjmax]);
}

__global__
void all_up_Kernel(float* uall,float* pall,
		      const float* u,const float* p,int n){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  uall[(x+d_cells[n].cimin)+d_sys.jmax*(y+d_cells[n].cjmin)]
    +=u[x+d_sys.cjmax*y+n*d_sys.cimax*d_sys.cjmax];
  pall[(x+d_cells[n].cimin)+d_sys.jmax*(y+d_cells[n].cjmin)]
    +=p[x+d_sys.cjmax*y+n*d_sys.cimax*d_sys.cjmax];
}


__global__
void init_normal_vector_field_Kernel(float* vx,float* vy){
//void init_normal_vector_field_Kernel(float** vx,float** vy){
//void init_normal_vector_field_Kernel(float** vx,float** vy,float** vz){
  int n = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  vx[n*d_sys.cimax*d_sys.cimax+x+d_sys.cimax*y]=0.0f;
  vy[n*d_sys.cimax*d_sys.cimax+x+d_sys.cimax*y]=0.0f;
  //vz[n][x+d_sys.cimax*y]=0.0f;
}

__global__
void init_normal_vector_field_all_Kernel(float* vx,float* vy){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.imax;
  int x = (index / d_sys.imax) % d_sys.jmax;

  vx[x+d_sys.imax*y]=0.0f;
  vy[x+d_sys.imax*y]=0.0f;
  //vz[n][x+d_sys.imax*y]=0.0f;
}

__global__
void normal_vector_field_Kernel(float* vx,float* vy,float* u){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % d_sys.cimax;
  int y = (index / d_sys.cimax) % d_sys.cjmax;
  int n = ((index / d_sys.cimax) / d_sys.cjmax) % d_cnum;

  vx[n*d_sys.cimax*d_sys.cimax+x+d_sys.cimax*y]=0.0f;
  vy[n*d_sys.cimax*d_sys.cimax+x+d_sys.cimax*y]=0.0f;

  if(x>0 && x<d_sys.cimax-1 && y>0 && y<d_sys.cimax-1){
    vx[n*d_sys.cimax*d_sys.cimax+x+d_sys.cimax*y]
      =(u[(x+1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cimax]
	-u[(x-1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cimax])*0.5f/d_sys.dx;
    vy[n*d_sys.cimax*d_sys.cimax+x+d_sys.cimax*y]
      =(u[x+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cimax]
	-u[x+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cimax])*0.5f/d_sys.dy;
    //vz[n][x+d_sys.cimax*y]=(u[n][x+d_sys.cimax*(y+1)]-u[n][x+d_sys.cimax*(y-1)])*0.50f/d_sys.dz;
  }

  float vabs=
    __fsqrt_rn(
	       vx[n*d_sys.cimax*d_sys.cimax+x+d_sys.cimax*y]
	       *vx[n*d_sys.cimax*d_sys.cimax+x+d_sys.cimax*y]
	       +vy[n*d_sys.cimax*d_sys.cimax+x+d_sys.cimax*y]
	       *vy[n*d_sys.cimax*d_sys.cimax+x+d_sys.cimax*y]
	       //+vz[n][x+d_sys.cimax*y]*vz[n][x+d_sys.cimax*y]
	       );
  if(vabs>0.00001f){
    vx[n*d_sys.cimax*d_sys.cimax+x+d_sys.cimax*y]/=vabs;
    vy[n*d_sys.cimax*d_sys.cimax+x+d_sys.cimax*y]/=vabs;
    //vz[n][x+d_sys.cimax*y]/=vabs;
  }
  else{
    vx[n*d_sys.cimax*d_sys.cimax+x+d_sys.cimax*y]=0.0f;
    vy[n*d_sys.cimax*d_sys.cimax+x+d_sys.cimax*y]=0.0f;
    //vz[n][x+d_sys.cimax*y]=0.0f;
  }
}

__global__
void normal_vector_field_all_Kernel(float* vx,float* vy,//float* vz,
				    float* pf){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.imax;
  int x = (index / d_sys.imax) % d_sys.jmax;

  vx[x+d_sys.imax*y]=0.0f;
  vy[x+d_sys.imax*y]=0.0f;

  if(x>0 && x<d_sys.imax-1 && y>0 && y<d_sys.jmax-1){
    vx[x+d_sys.imax*y]=(pf[(x+1)+d_sys.imax*y]-pf[(x-1)+d_sys.imax*y])*0.50f/d_sys.dx;
    vy[x+d_sys.imax*y]=(pf[x+d_sys.imax*(y+1)]-pf[x+d_sys.imax*(y-1)])*0.50f/d_sys.dy;
    //vz[n][x+d_sys.imax*y]=(pf[x+d_sys.imax*(y+1)]-pf[x+d_sys.imax*(y-1)])*0.50f/d_sys.dz;
  }

  float vabs=
    sqrtf(vx[x+d_sys.imax*y]*vx[x+d_sys.imax*y]+vy[x+d_sys.imax*y]*vy[x+d_sys.imax*y]
	  //+vz[x+d_sys.imax*y]*vz[x+d_sys.imax*y]
	  );
  if(vabs>0.0000f){
    vx[x+d_sys.imax*y]/=vabs;
    vy[x+d_sys.imax*y]/=vabs;
    //vz[x+d_sys.imax*y]/=vabs;
  }
  else{
    vx[x+d_sys.imax*y]=0.0f;
    vy[x+d_sys.imax*y]=0.0f;
    //vz[x+d_sys.imax*y]=0.0f;
  }
}


__global__
void normal_vector_field_all_2_Kernel(float* vx,float* vy,float* pf,int imin,int jmin,int imax,int jmax){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % imax;
  int y = (index / imax) % jmax;

  vx[(x+imin)+d_sys.imax*(y+jmin)]=0.0f;
  vy[(x+imin)+d_sys.imax*(y+jmin)]=0.0f;

  if(x+imin>0 && x+imin<d_sys.imax-1 && y+jmin>0 && y+jmin<d_sys.jmax-1){
    vx[(x+imin)+d_sys.imax*(y+jmin)]=(pf[(x+imin+1)+d_sys.imax*(y+jmin)]
				      -pf[(x+imin-1)+d_sys.imax*(y+jmin)])*0.5f/d_sys.dx;
    vy[(x+imin)+d_sys.imax*(y+jmin)]=(pf[(x+imin)+d_sys.imax*(y+jmin+1)]
				      -pf[(x+imin)+d_sys.imax*(y+jmin-1)])*0.5f/d_sys.dy;
  }

  float vabs=
    sqrtf(vx[(x+imin)+d_sys.imax*(y+jmin)]*vx[(x+imin)+d_sys.imax*(y+jmin)]
	  +vy[(x+imin)+d_sys.imax*(y+jmin)]*vy[(x+imin)+d_sys.imax*(y+jmin)]
	  //+vz[x+d_sys.imax*y]*vz[x+d_sys.imax*y]
	  );
  if(vabs>0.0000f){
    vx[(x+imin)+d_sys.imax*(y+jmin)]/=vabs;
    vy[(x+imin)+d_sys.imax*(y+jmin)]/=vabs;
    //vz[x+d_sys.imax*y]/=vabs;
  }
  else{
    vx[(x+imin)+d_sys.imax*(y+jmin)]=0.0f;
    vy[(x+imin)+d_sys.imax*(y+jmin)]=0.0f;
    //vz[x+d_sys.imax*y]=0.0f;
  }
}



__global__
void normal_vector_field_sc_Kernel(float* vx_s,float* vy_s,float* s,
				   float* vx_c,float* vy_c,float* c){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.imax;
  int x = (index / d_sys.imax) % d_sys.jmax;

  vx_s[x+d_sys.imax*y]=0.0f; vy_s[x+d_sys.imax*y]=0.0f;
  vx_c[x+d_sys.imax*y]=0.0f; vy_c[x+d_sys.imax*y]=0.0f;

  if(x>0 && x<d_sys.imax-1 && y>0 && y<d_sys.jmax-1){
    vx_s[x+d_sys.imax*y]=(s[(x+1)+d_sys.imax*y]-s[(x-1)+d_sys.imax*y])*0.50f/d_sys.dx;
    vy_s[x+d_sys.imax*y]=(s[x+d_sys.imax*(y+1)]-s[x+d_sys.imax*(y-1)])*0.50f/d_sys.dy;
    vx_c[x+d_sys.imax*y]=(c[(x+1)+d_sys.imax*y]-c[(x-1)+d_sys.imax*y])*0.50f/d_sys.dx;
    vy_c[x+d_sys.imax*y]=(c[x+d_sys.imax*(y+1)]-c[x+d_sys.imax*(y-1)])*0.50f/d_sys.dy;
    //vz[n][x+d_sys.imax*y]=(pf[x+d_sys.imax*(y+1)]-pf[x+d_sys.imax*(y-1)])*0.50f/d_sys.dz;
  }

  float vabs_s=sqrtf(vx_s[x+d_sys.imax*y]*vx_s[x+d_sys.imax*y]
		     +vy_s[x+d_sys.imax*y]*vy_s[x+d_sys.imax*y]
		     //+vz[x+d_sys.imax*y]*vz[x+d_sys.imax*y]
		     );
  float vabs_c=sqrtf(vx_c[x+d_sys.imax*y]*vx_c[x+d_sys.imax*y]
		     +vy_c[x+d_sys.imax*y]*vy_c[x+d_sys.imax*y]
		     //+vz[x+d_sys.imax*y]*vz[x+d_sys.imax*y]
		     );

  if(vabs_s>0.0000f){
    vx_s[x+d_sys.imax*y]/=vabs_s; vy_s[x+d_sys.imax*y]/=vabs_s; //vz[x+d_sys.imax*y]/=vabs;
  }
  else{
    vx_s[x+d_sys.imax*y]=0.0f; vy_s[x+d_sys.imax*y]=0.0f; //vz[x+d_sys.imax*y]=0.0f;
  }

  if(vabs_c>0.0000f){
    vx_c[x+d_sys.imax*y]/=vabs_c; vy_c[x+d_sys.imax*y]/=vabs_c; //vz[x+d_sys.imax*y]/=vabs;
  }
  else{
    vx_c[x+d_sys.imax*y]=0.0f; vy_c[x+d_sys.imax*y]=0.0f; //vz[x+d_sys.imax*y]=0.0f;
  }
}

__global__
void normal_vector_field_sc_2_Kernel(float* vx_s,float* vy_s,float* s,
				     float* vx_c,float* vy_c,float* c,
				     int imin,int jmin,int imax,int jmax){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % imax;
  int y = (index / imax) % jmax;

  vx_s[(x+imin)+d_sys.imax*(y+jmin)]=0.0f; vy_s[(x+imin)+d_sys.imax*(y+jmin)]=0.0f;
  vx_c[(x+imin)+d_sys.imax*(y+jmin)]=0.0f; vy_c[(x+imin)+d_sys.imax*(y+jmin)]=0.0f;

  if(x+imin>0 && x+imin<d_sys.imax-1 && y+jmin>0 && y+jmin<d_sys.jmax-1){
    vx_s[(x+imin)+d_sys.imax*(y+jmin)]=(s[(x+imin+1)+d_sys.imax*(y+jmin)]
					-s[(x+imin-1)+d_sys.imax*(y+jmin)])*0.5f/d_sys.dx;
    vy_s[(x+imin)+d_sys.imax*(y+jmin)]=(s[(x+imin)+d_sys.imax*(y+jmin+1)]
					-s[(x+imin)+d_sys.imax*(y+jmin-1)])*0.5f/d_sys.dy;
    vx_c[(x+imin)+d_sys.imax*(y+jmin)]=(c[(x+imin+1)+d_sys.imax*(y+jmin)]
					-c[(x+imin-1)+d_sys.imax*(y+jmin)])*0.5f/d_sys.dx;
    vy_c[(x+imin)+d_sys.imax*(y+jmin)]=(c[(x+imin)+d_sys.imax*(y+jmin+1)]
					-c[(x+imin)+d_sys.imax*(y+jmin-1)])*0.5f/d_sys.dy;
    //vz[n][x+d_sys.imax*y]=(pf[x+d_sys.imax*(y+1)]-pf[x+d_sys.imax*(y-1)])*0.50f/d_sys.dz;
  }

  float vabs_s=sqrtf(vx_s[(x+imin)+d_sys.imax*(y+jmin)]*vx_s[(x+imin)+d_sys.imax*(y+jmin)]
		     +vy_s[(x+imin)+d_sys.imax*(y+jmin)]*vy_s[(x+imin)+d_sys.imax*(y+jmin)]
		     //+vz[x+d_sys.imax*y]*vz[x+d_sys.imax*y]
		     );
  float vabs_c=sqrtf(vx_c[(x+imin)+d_sys.imax*(y+jmin)]*vx_c[(x+imin)+d_sys.imax*(y+jmin)]
		     +vy_c[(x+imin)+d_sys.imax*(y+jmin)]*vy_c[(x+imin)+d_sys.imax*(y+jmin)]
		     //+vz[x+d_sys.imax*y]*vz[x+d_sys.imax*y]
		     );

  if(vabs_s>0.0000f){
    vx_s[(x+imin)+d_sys.imax*(y+jmin)]/=vabs_s;
    vy_s[(x+imin)+d_sys.imax*(y+jmin)]/=vabs_s; //vz[x+d_sys.imax*y]/=vabs;
  }
  else{
    vx_s[(x+imin)+d_sys.imax*(y+jmin)]=0.0f;
    vy_s[(x+imin)+d_sys.imax*(y+jmin)]=0.0f; //vz[x+d_sys.imax*y]=0.0f;
  }

  if(vabs_c>0.0000f){
    vx_c[(x+imin)+d_sys.imax*(y+jmin)]/=vabs_c;
    vy_c[(x+imin)+d_sys.imax*(y+jmin)]/=vabs_c; //vz[x+d_sys.imax*y]/=vabs;
  }
  else{
    vx_c[(x+imin)+d_sys.imax*(y+jmin)]=0.0f;
    vy_c[(x+imin)+d_sys.imax*(y+jmin)]=0.0f; //vz[x+d_sys.imax*y]=0.0f;
  }
}




__global__
void boundary_normal_vector_field_Kernel(float* vx,float* vy
//void boundary_normal_vector_field_Kernel(float** vx,float** vy
					 //,float** vz
					 ){
  int n = blockIdx.y;
  int y = blockIdx.x * blockDim.x + threadIdx.x;

  vx[n*d_sys.cimax*d_sys.cimax+0+d_sys.cimax*y]
    =vx[n*d_sys.cimax*d_sys.cimax+(d_sys.cimax-2)+d_sys.cimax*y];
  vy[n*d_sys.cimax*d_sys.cimax+0+d_sys.cimax*y]
    =vy[n*d_sys.cimax*d_sys.cimax+(d_sys.cimax-2)+d_sys.cimax*y];
  // vz[n][x+d_sys.cimax*y]=vz[n][(d_sys.cimax-2)+d_sys.cimax*y];
  vx[n*d_sys.cimax*d_sys.cimax+y+d_sys.cimax*0]
    =vx[n*d_sys.cimax*d_sys.cimax+y+d_sys.cimax*(d_sys.cimax-2)];
  vy[n*d_sys.cimax*d_sys.cimax+y+d_sys.cimax*0]
    =vy[n*d_sys.cimax*d_sys.cimax+y+d_sys.cimax*(d_sys.cimax-2)];
  //vz[n][x+d_sys.cimax*y]=vz[n][x+d_sys.cimax*(d_sys.cjmax-2)];

  vx[n*d_sys.cimax*d_sys.cimax+d_sys.cimax-1+d_sys.cimax*y]
    =vx[n*d_sys.cimax*d_sys.cimax+(d_sys.cimax-2)+d_sys.cimax*y];
  vy[n*d_sys.cimax*d_sys.cimax+d_sys.cimax-1+d_sys.cimax*y]
    =vy[n*d_sys.cimax*d_sys.cimax+(d_sys.cimax-2)+d_sys.cimax*y];
  // vz[n][x+d_sys.cimax*y]=vz[n][(d_sys.cimax-2)+d_sys.cimax*y];
  vx[n*d_sys.cimax*d_sys.cimax+y+d_sys.cimax*(d_sys.cimax-1)]
    =vx[n*d_sys.cimax*d_sys.cimax+y+d_sys.cimax*1];
  vy[n*d_sys.cimax*d_sys.cimax+y+d_sys.cimax*(d_sys.cimax-1)]
    =vy[n*d_sys.cimax*d_sys.cimax+y+d_sys.cimax*1];
  //vz[n][x+d_sys.cimax*y]=vz[n][x+d_sys.cimax*(d_sys.cjmax-2)];

}

__global__
void boundary_normal_vector_field_all_Kernel(float* vx,float* vy
//void boundary_normal_vector_field_Kernel(float** vx,float** vy
					 //,float** vz
					 ){
  //int n = blockIdx.y;
  int y = blockIdx.x * blockDim.x + threadIdx.x;

  vx[0+d_sys.imax*y]=vx[(d_sys.imax-2)+d_sys.imax*y];
  vy[0+d_sys.imax*y]=vy[(d_sys.imax-2)+d_sys.imax*y];
  // vz[n][x+d_sys.imax*y]=vz[n][(d_sys.imax-2)+d_sys.imax*y];
  vx[y+d_sys.imax*0]=vx[y+d_sys.imax*(d_sys.imax-2)];
  vy[y+d_sys.imax*0]=vy[y+d_sys.imax*(d_sys.imax-2)];
  //vz[n][x+d_sys.imax*y]=vz[n][x+d_sys.imax*(d_sys.jmax-2)];

  vx[d_sys.imax-1+d_sys.imax*y]=vx[(d_sys.imax-2)+d_sys.imax*y];
  vy[d_sys.imax-1+d_sys.imax*y]=vy[(d_sys.imax-2)+d_sys.imax*y];
  // vz[n][x+d_sys.imax*y]=vz[n][(d_sys.imax-2)+d_sys.imax*y];
  vx[y+d_sys.imax*(d_sys.imax-1)]=vx[y+d_sys.imax*1];
  vy[y+d_sys.imax*(d_sys.imax-1)]=vy[y+d_sys.imax*1];
  //vz[n][x+d_sys.imax*y]=vz[n][x+d_sys.imax*(d_sys.jmax-2)];
}

__global__
void boundary_normal_vector_field_sc_Kernel(float* vx_s,float* vy_s, //float* vz,
					     float* vx_c,float* vy_c //float* vz,
					     ){
  int y = blockIdx.x * blockDim.x + threadIdx.x;

  vx_s[0+d_sys.imax*y]=vx_s[(d_sys.imax-2)+d_sys.imax*y];
  vy_s[0+d_sys.imax*y]=vy_s[(d_sys.imax-2)+d_sys.imax*y];
  // vz_s[n][x+d_sys.imax*y]=vz_s[n][(d_sys.imax-2)+d_sys.imax*y];
  vx_s[y+d_sys.imax*0]=vx_s[y+d_sys.imax*(d_sys.imax-2)];
  vy_s[y+d_sys.imax*0]=vy_s[y+d_sys.imax*(d_sys.imax-2)];
  //vz_s[n][x+d_sys.imax*y]=vz_s[n][x+d_sys.imax*(d_sys.jmax-2)];

  vx_s[d_sys.imax-1+d_sys.imax*y]=vx_s[(d_sys.imax-2)+d_sys.imax*y];
  vy_s[d_sys.imax-1+d_sys.imax*y]=vy_s[(d_sys.imax-2)+d_sys.imax*y];
  // vz_s[n][x+d_sys.imax*y]=vz_s[n][(d_sys.imax-2)+d_sys.imax*y];
  vx_s[y+d_sys.imax*(d_sys.imax-1)]=vx_s[y+d_sys.imax*1];
  vy_s[y+d_sys.imax*(d_sys.imax-1)]=vy_s[y+d_sys.imax*1];
  //vz_s[n][x+d_sys.imax*y]=vz_s[n][x+d_sys.imax*(d_sys.jmax-2)];

  vx_c[0+d_sys.imax*y]=vx_c[(d_sys.imax-2)+d_sys.imax*y];
  vy_c[0+d_sys.imax*y]=vy_c[(d_sys.imax-2)+d_sys.imax*y];
  // vz_c[n][x+d_sys.imax*y]=vz_c[n][(d_sys.imax-2)+d_sys.imax*y];
  vx_c[y+d_sys.imax*0]=vx_c[y+d_sys.imax*(d_sys.imax-2)];
  vy_c[y+d_sys.imax*0]=vy_c[y+d_sys.imax*(d_sys.imax-2)];
  //vz_c[n][x+d_sys.imax*y]=vz_c[n][x+d_sys.imax*(d_sys.jmax-2)];

  vx_c[d_sys.imax-1+d_sys.imax*y]=vx_c[(d_sys.imax-2)+d_sys.imax*y];
  vy_c[d_sys.imax-1+d_sys.imax*y]=vy_c[(d_sys.imax-2)+d_sys.imax*y];
  // vz_c[n][x+d_sys.imax*y]=vz_c[n][(d_sys.imax-2)+d_sys.imax*y];
  vx_c[y+d_sys.imax*(d_sys.imax-1)]=vx_c[y+d_sys.imax*1];
  vy_c[y+d_sys.imax*(d_sys.imax-1)]=vy_c[y+d_sys.imax*1];
  //vz_c[n][x+d_sys.imax*y]=vz_c[n][x+d_sys.imax*(d_sys.jmax-2)];

}



__global__
void reshaping_Kernel(float* vx,float* vy, //const float** vz,
 		      float* u,float* u_temp,float dtau){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % d_sys.cimax;
  int y = (index / d_sys.cimax) % d_sys.cjmax;
  int n = ((index / d_sys.cimax) / d_sys.cjmax) % d_cnum;

  if(x>0 && x<d_sys.cimax-1 && y>0 && y<d_sys.cimax-1){
    u_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cimax]
      =u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cimax]
      +dtau*d_para.D_u*laplacian8(u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax],
				  u[(x-1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax],
				  u[(x+1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax],
				  u[x+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax],
				  u[x+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax],
				  u[(x-1)+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax],
				  u[(x+1)+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cjmax],
				  u[(x-1)+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax],
				  u[(x+1)+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cjmax])
      // +dtau*d_para.D_u*laplacian4(u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cimax],
      // 	     			  u[(x-1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cimax],
      // 				  u[(x+1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cimax],
      // 	     			  u[x+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cimax],
      // 				  u[x+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cimax])
      -dtau*sqrtf(2.0f*d_para.D_u)*
      (
       (u[(x+1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cimax]
	*(1.0f-u[(x+1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cimax])
	*vx[(x+1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cimax]
	-u[(x-1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cimax]
	*(1.0f-u[(x-1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cimax])
	*vx[(x-1)+d_sys.cimax*y+n*d_sys.cimax*d_sys.cimax])
       *0.5f/d_sys.dx
       +(u[x+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cimax]
	 *(1.0f-u[x+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cimax])
	 *vy[x+d_sys.cimax*(y+1)+n*d_sys.cimax*d_sys.cimax]
	 -u[x+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cimax]
	 *(1.0f-u[x+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cimax])
	 *vy[x+d_sys.cimax*(y-1)+n*d_sys.cimax*d_sys.cimax])
       *0.5f/d_sys.dy
       )
      ;
  }
}


__global__
void reshaping_all_Kernel(float* vx,float* vy, //const float** vz,
			  float* pf,float* pf_temp,float dtau,float D){
// void reshaping_Kernel(float** vx,float** vy, //const float** vz,
// 		      float** u,float** u_temp,float dtau){
  //int n = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.imax;
  int x = (index / d_sys.imax) % d_sys.jmax;

  if(x>0 && x<d_sys.imax-1 && y>0 && y<d_sys.imax-1){
    pf_temp[x+d_sys.imax*y]
      =pf[x+d_sys.imax*y]
      +dtau*D
      //*laplacian4(i,j)
      *(
	(pf[x+d_sys.imax*(y-1)]-2.00f*pf[x+d_sys.imax*y]+pf[x+d_sys.imax*(y+1)])/(d_sys.dx*d_sys.dx)
      	+(pf[(x-1)+d_sys.imax*y]-2.00f*pf[x+d_sys.imax*y]+pf[(x+1)+d_sys.imax*y])/(d_sys.dy*d_sys.dy)
	)
      // *(
      // 	(pf[x+d_sys.imax*(y-1)]-2.00f*pf[x+d_sys.imax*y]+pf[x+d_sys.imax*(y+1)])/(d_sys.dx*d_sys.dx)
      // 	+(pf[(x-1)+d_sys.imax*y]-2.00f*pf[x+d_sys.imax*y]+pf[(x+1)+d_sys.imax*y])/(d_sys.dy*d_sys.dy)
      // 	+(0.50f*(pf[(x-1)+d_sys.imax*(y-1)]+pf[(x-1)+d_sys.imax*(y+1)]
      // 		 +pf[(x+1)+d_sys.imax*(y-1)]+pf[(x+1)+d_sys.imax*(y+1)])
      // 	  -2.0f*pf[x+d_sys.imax*y])/(d_sys.dx*d_sys.dy)
      // 	)*0.50f
      -dtau*sqrtf(2.0f*D)*
      (
       (pf[(x+1)+d_sys.imax*y]*(1.0f-pf[(x+1)+d_sys.imax*y])*vx[(x+1)+d_sys.imax*y]
	-pf[(x-1)+d_sys.imax*y]*(1.0f-pf[(x-1)+d_sys.imax*y])*vx[(x-1)+d_sys.imax*y])
       *0.50f/d_sys.dx
       +(pf[x+d_sys.imax*(y+1)]*(1.0f-pf[x+d_sys.imax*(y+1)])*vy[x+d_sys.imax*(y+1)]
	 -pf[x+d_sys.imax*(y-1)]*(1.0f-pf[x+d_sys.imax*(y-1)])*vy[x+d_sys.imax*(y-1)])
       *0.50f/d_sys.dy
       )
      ;
  }
}

__global__
void reshaping_all_2_Kernel(float* vx,float* vy, //const float** vz,
			    float* pf,float* pf_temp,float dtau,float D,
			    int imin,int jmin,int imax,int jmax){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % imax;
  int y = (index / imax) % jmax;

  if(x+imin>0 && x+imin<d_sys.imax-1 && y+jmin>0 && y+jmin<d_sys.imax-1){
    pf_temp[(x+imin)+d_sys.imax*(y+jmin)]
      =pf[(x+imin)+d_sys.imax*(y+jmin)]
      +dtau*D
      *laplacian8(pf[(x+imin)+d_sys.imax*(y+jmin)],
		  pf[(x+imin-1)+d_sys.imax*(y+jmin)],
		  pf[(x+imin+1)+d_sys.imax*(y+jmin)],
		  pf[(x+imin)+d_sys.imax*(y+jmin-1)],
		  pf[(x+imin)+d_sys.imax*(y+jmin+1)],
		  pf[(x+imin-1)+d_sys.imax*(y+jmin-1)],
		  pf[(x+imin+1)+d_sys.imax*(y+jmin-1)],
		  pf[(x+imin-1)+d_sys.imax*(y+jmin+1)],
		  pf[(x+imin+1)+d_sys.imax*(y+jmin+1)])
      // //*laplacian4(i,j)
      // +dtau*D
      // *(
      // 	(pf[(x+imin-1)+d_sys.imax*(y+jmin)]
      // 	 -2.0f*pf[(x+imin)+d_sys.imax*(y+jmin)]
      // 	 +pf[(x+imin+1)+d_sys.imax*(y+jmin)])/(d_sys.dx*d_sys.dx)
      // 	+(pf[(x+imin)+d_sys.imax*(y+jmin-1)]
      // 	  -2.0f*pf[(x+imin)+d_sys.imax*(y+jmin)]
      // 	  +pf[(x+imin)+d_sys.imax*(y+jmin+1)])/(d_sys.dy*d_sys.dy)
      // 	)
      -dtau*sqrtf(2.0f*D)*
      (
       (pf[(x+imin+1)+d_sys.imax*(y+jmin)]*(1.0f-pf[(x+imin+1)+d_sys.imax*(y+jmin)])
	*vx[(x+imin+1)+d_sys.imax*(y+jmin)]
	-pf[(x+imin-1)+d_sys.imax*(y+jmin)]*(1.0f-pf[(x+imin-1)+d_sys.imax*(y+jmin)])
	*vx[(x+imin-1)+d_sys.imax*(y+jmin)])
       *0.5f/d_sys.dx
       +(pf[(x+imin)+d_sys.imax*(y+jmin+1)]*(1.0f-pf[(x+imin)+d_sys.imax*(y+jmin+1)])
	 *vy[(x+imin)+d_sys.imax*(y+jmin+1)]
	 -pf[(x+imin)+d_sys.imax*(y+jmin-1)]*(1.0f-pf[(x+imin)+d_sys.imax*(y+jmin-1)])
	 *vy[(x+imin)+d_sys.imax*(y+jmin-1)])
       *0.5f/d_sys.dy
       )
      ;
  }
}



__global__
void u_temp_Kernel(float* u,float** u_temp){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  u[x+d_sys.cimax*y]=u_temp[0][x+d_sys.cimax*y];
}

__global__
void dev_Kernel(float*dev,float* u,float* u_temp){

  int n = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  dev[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cimax]
   =fabsf(u_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cimax]
	  -u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cimax]);
}

__global__
void dev2_Kernel(float*dev,float* u,float* u_temp){

  int n = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  dev[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]
    =u_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cimax]
    -u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax];
}


__global__
void dev_all_Kernel(float*dev,float* pf,float* pf_temp){

  //int n = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.imax;
  int x = (index / d_sys.imax) % d_sys.jmax;

  //  dev[x+d_sys.imax*y]
  dev[x+d_sys.imax*y]=fabsf(pf_temp[x+d_sys.imax*y]-pf[x+d_sys.imax*y]);
}

__global__
void dev_all_2_Kernel(float*dev,float* pf,float* pf_temp,int imin,int jmin,int imax,int jmax){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % imax;
  int y = (index / imax) % jmax;

  dev[(x+imin)+d_sys.imax*(y+jmin)]=fabsf(pf_temp[(x+imin)+d_sys.imax*(y+jmin)]-pf[(x+imin)+d_sys.imax*(y+jmin)]);
}


__global__
void dev_i_all_Kernel(int*dev_i,float* pf,float* pf_temp,int fi){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.imax;
  int x = (index / d_sys.imax) % d_sys.jmax;

  dev_i[x+d_sys.imax*y]=(int)(fabsf(pf_temp[x+d_sys.imax*y]
				       -pf[x+d_sys.imax*y])*fi);
  dev_i[x+d_sys.imax*y]=dev_i[x+d_sys.imax*y]*fi;
}



__global__
void d_dev_Kernel(float*dev,float* dd_dev,int n){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  dd_dev[x+d_sys.cimax*y]=dev[n*d_sys.cimax*d_sys.cimax+x+d_sys.cimax*y];
}

__global__
void init_medium_Kernel(float*d_c,float*d_uall){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.imax;
  int x = (index / d_sys.imax) % d_sys.jmax;

  d_c[x+d_sys.jmax*y]=1.0f-d_uall[x+d_sys.jmax*y];
  if(d_c[x+d_sys.jmax*y]<0.0f) d_c[x+d_sys.jmax*y]=0.0f;
  //if(d_c[x+d_sys.jmax*y]>1.0f) d_c[x+d_sys.jmax*y]=1.0f;
}


__global__
void boundary_dirichlet_Kernel(float*v,float*v_temp){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.imax;

  v[0+d_sys.imax*y]=1.0f; v[y+d_sys.imax*0]=1.0f;
  v[d_sys.imax-1+d_sys.imax*y]=1.0f; v[y+d_sys.imax*(d_sys.jmax-1)]=1.0f;
  v_temp[0+d_sys.imax*y]=1.0f; v_temp[y+d_sys.imax*0]=1.0f;
  v_temp[d_sys.imax-1+d_sys.imax*y]=1.0f; v_temp[y+d_sys.imax*(d_sys.jmax-1)]=1.0f;
}


__device__
float distance(float r1x,float r1y,float r2x,float r2y){
  return sqrtf((r1x-r2x)*(r1x-r2x)+(r1y-r2y)*(r1y-r2y));
}


__global__
void copy_mother_cell_Kernel(float*u,float*m_u,int m){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  m_u[x+d_sys.cimax*y]=u[x+d_sys.cimax*y+m*d_sys.cimax*d_sys.cjmax];
}

__global__
void set_daughter_cells_Kernel(float*u,float*u_temp,
			       float r1x,float r1y, float r2x,float r2y,
			       int m,int d2){
  int n = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  if(n==m){
    float rx,ry;
    rx=d_cells[m].Gx-d_sys.cimax*0.5*d_sys.dx+(x+0.5)*d_sys.dx;
    ry=d_cells[m].Gy-d_sys.cjmax*0.5*d_sys.dy+(y+0.5)*d_sys.dy;
    //printf("%lf\t%lf\n",rx,ry);

    float d=distance(r1x,r1y,r2x,r2y);
    float g=(r1x-r2x)/d*(rx-(r1x+r2x)*0.5f)+(r1y-r2y)/d*(ry-(r1y+r2y)*0.5f);
    float chi=0.5f*(1+tanhf(g/d_para.ep_d));
    u_temp[x+d_sys.cimax*y+d2*d_sys.cimax*d_sys.cjmax]=u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]*(1-chi);
    u_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]=u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]*chi;
  }
  else{
    u_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]=u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax];
  }
}

__global__
void set_daughter_cells_2_Kernel(float*u,float*u_temp,float*p,float*p_temp,
				 float r1x,float r1y, float r2x,float r2y,
				 int m,int d2){
  int n = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;

  if(n==m){
    float rx,ry;
    rx=d_cells[m].Gx-d_sys.cimax*0.5*d_sys.dx+(x+0.5)*d_sys.dx;
    ry=d_cells[m].Gy-d_sys.cjmax*0.5*d_sys.dy+(y+0.5)*d_sys.dy;
    //printf("%lf\t%lf\n",rx,ry);

    float d=distance(r1x,r1y,r2x,r2y);
    float g=(r1x-r2x)/d*(rx-(r1x+r2x)*0.5f)+(r1y-r2y)/d*(ry-(r1y+r2y)*0.5f);
    float chi=0.5f*(1+tanhf(g/d_para.ep_d));
    u_temp[x+d_sys.cimax*y+d2*d_sys.cimax*d_sys.cjmax]=u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]*(1-chi);
    u_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]=u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]*chi;
    p_temp[x+d_sys.cimax*y+d2*d_sys.cimax*d_sys.cjmax]=p[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]*(1-chi);
    p_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]=p[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]*chi;
  }
  else{
    u_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]=u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax];
    p_temp[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]=p[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax];
  }
}


__global__
void force_field_Kernel(float*m_u,float*e_eta,
			float r1x,float r1y,float r2x,float r2y,
			float*f1x,float*f1y,
			float*f2x,float*f2y,
			int n){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;


  f1x[x+d_sys.cimax*y]=0.0f;
  f1y[x+d_sys.cimax*y]=0.0f;
  f2x[x+d_sys.cimax*y]=0.0f;
  f2y[x+d_sys.cimax*y]=0.0f;

  float rx,ry;
  rx=(float)(d_cells[n].Gx-d_sys.cimax*0.5f*d_sys.dx+(x+0.5f)*d_sys.dx);
  ry=(float)(d_cells[n].Gy-d_sys.cjmax*0.5f*d_sys.dy+(y+0.5f)*d_sys.dy);

  if(distance(r1x,r1y,rx,ry)<distance(r2x,r2y,rx,ry)){
    f1x[x+d_sys.cimax*y]=
      (d_para.rho0-d_para.rhoe*e_eta[(x+d_cells[n].cimin)+d_sys.jmax*(y+d_cells[n].cjmin)])
      *m_u[x+d_sys.cimax*y]*(1.0f-m_u[x+d_sys.cimax*y])*(rx-r1x);
    f1y[x+d_sys.cimax*y]=
      (d_para.rho0-d_para.rhoe*e_eta[(x+d_cells[n].cimin)+d_sys.jmax*(y+d_cells[n].cjmin)])
      *m_u[x+d_sys.cimax*y]*(1.0f-m_u[x+d_sys.cimax*y])*(ry-r1y);
  }
  if(distance(r1x,r1y,rx,ry)>distance(r2x,r2y,rx,ry)){
    f2x[x+d_sys.cimax*y]=
      (d_para.rho0-d_para.rhoe*e_eta[(x+d_cells[n].cimin)+d_sys.jmax*(y+d_cells[n].cjmin)])
      *m_u[x+d_sys.cimax*y]*(1.0f-m_u[x+d_sys.cimax*y])*(rx-r2x);
    f2y[x+d_sys.cimax*y]=
      (d_para.rho0-d_para.rhoe*e_eta[(x+d_cells[n].cimin)+d_sys.jmax*(y+d_cells[n].cjmin)])
      *m_u[x+d_sys.cimax*y]*(1.0f-m_u[x+d_sys.cimax*y])*(ry-r2y);
  }

  //printf("%lf\t%lf\n",f1x[x+d_sys.cimax*y],f1y[x+d_sys.cimax*y]);

}

__global__
void set_seed_p_Kernel(float*p,float Pcx,float Pcy,int n){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;


  float r
    =sqrtf((d_sys.dx*(x+d_cells[n].cimin+0.5f)-Pcx)*(d_sys.dx*(x+d_cells[n].cimin+0.5f)-Pcx)
	   +(d_sys.dy*(y+d_cells[n].cjmin+0.5f)-Pcy)*(d_sys.dy*(y+d_cells[n].cjmin+0.5f)-Pcy));

  //if(r<d_para.p_r) p[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]=1.0f;
  p[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]+=(1.0f-tanhf((r-d_para.p_r)/(sqrtf(2.0f*d_para.D_p))))*0.5f;
  if(p[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]>1.0f)
    p[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]=1.0f;
}

__global__
void set_seed_s_Kernel(float*u,float*s,float Pcx,float Pcy,int n){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % d_sys.cimax;
  int y = (index / d_sys.cimax) % d_sys.cjmax;


  float r
    =sqrtf((d_sys.dx*(x+d_cells[n].cimin+0.5f)-Pcx)*(d_sys.dx*(x+d_cells[n].cimin+0.5f)-Pcx)
	   +(d_sys.dy*(y+d_cells[n].cjmin+0.5f)-Pcy)*(d_sys.dy*(y+d_cells[n].cjmin+0.5f)-Pcy));

  float seed
    =(1.0f-tanhf((r-d_para.p_r)/(sqrtf(2.0f*d_para.D_u))))*0.5f;
  if(seed>0.00001f){
    s[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]+=seed;
    if(s[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]>1.0000f)
      s[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]=1.0f;
    u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]-=seed;
    if(u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]<0.0000001f)
      u[x+d_sys.cimax*y+n*d_sys.cimax*d_sys.cjmax]=0.0f;
  }

}

__global__
void overlap_us_Kernel(float *u_all,float *s,float *overlap){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.imax;
  int x = (index / d_sys.imax) % d_sys.jmax;

  overlap[x+d_sys.imax*y]=s[x+d_sys.imax*y]*(1.0f-s[x+d_sys.imax*y])*u_all[x+d_sys.imax*y]*d_sys.dx*d_sys.dy;
}


__global__
void debug_phi_Kernel(float* phim,float** u,int n){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int y = index % d_sys.cimax;
  int x = (index / d_sys.cimax) % d_sys.cjmax;
  phim[(x+d_cells[n].cimin)+d_sys.imax*(y+d_cells[n].cjmin)]-=h(u[n][x+d_sys.cimax*y]);
}


/////////////////////////////////////////////////////////////////////////////////
//
//  function code
//
/////////////////////////////////////////////////////////////////////////////////

float distance_h(float (&r1)[2],float (&r2)[2]){
  return sqrt((r1[0]-r2[0])*(r1[0]-r2[0])+(r1[1]-r2[1])*(r1[1]-r2[1]));
}


float volume_h(float*d_u){
  float result;

  int n=CIMAX*CJMAX;
  int th=WARP;
  //int blocks=(n-1)/th+1;  //reduce0
  int blocks=(n-1)/(2*th)+1;  //reduce3

  int shared_mem_size=2*th*sizeof(float);

  float *d1_out, *d2_out;
  cudaMalloc((void**)&d1_out,sizeof(float)*blocks);
  cudaMalloc((void**)&d2_out,sizeof(float)*blocks);

  float *dd_u;
  cudaMalloc((void**)&dd_u,sizeof(float)*n);
  h_Kernel<<<n/th,th>>>(dd_u,d_u);

  float **in=&dd_u,**out=&d1_out;
  while (blocks>1) {
    //reduce0<<<blocks, th, shared_mem_size>>>(*in, *out, n);
    reduce3<<<blocks, th, shared_mem_size>>>(*in, *out, n);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
      puts(cudaGetErrorString(err));
    }
    if(*out==d1_out){
      out=&d2_out; in=&d1_out;
    }
    else{
      out=&d1_out; in=&d2_out;
    }
    n=blocks;
    // blocks=(blocks-1)/th+1;  //reduce0
    blocks=(blocks-1)/(2*th)+1;  //reduce3
    cudaDeviceSynchronize();
  }
  //reduce0<<<blocks,th,shared_mem_size>>>(*in,*out,n);
  reduce3<<<blocks,th,shared_mem_size>>>(*in,*out,n);
  cudaMemcpy(&result,*out,sizeof(float),cudaMemcpyDeviceToHost);

  cudaFree(d1_out);
  cudaFree(d2_out);
  d1_out=NULL;
  d2_out=NULL;
  cudaFree(dd_u);
  dd_u=NULL;

  return result*DX*DY;
  //return result;
}



float volume(float*d_u){
  float result;

  int n=CIMAX*CJMAX;
  int th=WARP;
  //int blocks=(n-1)/th+1;  //reduce0
  int blocks=(n-1)/(2*th)+1;  //reduce3

  int shared_mem_size=2*th*sizeof(float);

  float *d1_out, *d2_out;
  cudaMalloc((void**)&d1_out,sizeof(float)*blocks);
  cudaMalloc((void**)&d2_out,sizeof(float)*blocks);

  float **in=&d_u,**out=&d1_out;
  while (blocks>1) {
    //reduce0<<<blocks, th, shared_mem_size>>>(*in, *out, n);
    reduce3<<<blocks, th, shared_mem_size>>>(*in, *out, n);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
      puts(cudaGetErrorString(err));
    }
    if(*out==d1_out){
      out=&d2_out; in=&d1_out;
    }
    else{
      out=&d1_out; in=&d2_out;
    }
    n=blocks;
    // blocks=(blocks-1)/th+1;  //reduce0
    blocks=(blocks-1)/(2*th)+1;  //reduce3
    cudaDeviceSynchronize();
  }
  //reduce0<<<blocks,th,shared_mem_size>>>(*in,*out,n);
  reduce3<<<blocks,th,shared_mem_size>>>(*in,*out,n);
  cudaMemcpy(&result,*out,sizeof(float),cudaMemcpyDeviceToHost);

  cudaFree(d1_out);
  cudaFree(d2_out);
  d1_out=NULL;
  d2_out=NULL;

  return result*DX*DY;
}

float volume_all(float*d_u){
  float result;

  int n=IMAX*JMAX;
  int th=WARP;
  //int blocks=(n-1)/th+1;  //reduce0
  int blocks=(n-1)/(2*th)+1;  //reduce3

  int shared_mem_size=2*th*sizeof(float);

  float *d1_out, *d2_out;
  cudaMalloc((void**)&d1_out,sizeof(float)*blocks);
  cudaMalloc((void**)&d2_out,sizeof(float)*blocks);

  float **in=&d_u,**out=&d1_out;
  while (blocks>1) {
    //reduce0<<<blocks, th, shared_mem_size>>>(*in, *out, n);
    reduce3<<<blocks, th, shared_mem_size>>>(*in, *out, n);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
      puts(cudaGetErrorString(err));
    }
    if(*out==d1_out){
      out=&d2_out; in=&d1_out;
    }
    else{
      out=&d1_out; in=&d2_out;
    }
    n=blocks;
    // blocks=(blocks-1)/th+1;  //reduce0
    blocks=(blocks-1)/(2*th)+1;  //reduce3
    cudaDeviceSynchronize();
  }
  //reduce0<<<blocks,th,shared_mem_size>>>(*in,*out,n);
  reduce3<<<blocks, th, shared_mem_size>>>(*in,*out,n);
  cudaMemcpy(&result,*out,sizeof(float),cudaMemcpyDeviceToHost);

  cudaFree(d1_out);
  cudaFree(d2_out);
  d1_out=NULL;
  d2_out=NULL;

  return result*DX*DY;
}

int volume_i_all(int*d_u){
  int result;

  int n=IMAX*JMAX;
  int th=WARP;
  //int blocks=(n-1)/th+1;  //reduce0
  int blocks=(n-1)/(2*th)+1;  //reduce3

  int shared_mem_size=2*th*sizeof(int);

  int *d1_out, *d2_out;
  cudaMalloc((void**)&d1_out,sizeof(int)*blocks);
  cudaMalloc((void**)&d2_out,sizeof(int)*blocks);

  int **in=&d_u,**out=&d1_out;
  while (blocks>1) {
    reduce3_i<<<blocks, th, shared_mem_size>>>(*in, *out, n);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
      puts(cudaGetErrorString(err));
    }
    if(*out==d1_out){
      out=&d2_out; in=&d1_out;
    }
    else{
      out=&d1_out; in=&d2_out;
    }
    n=blocks;
    //blocks=(blocks-1)/th+1;  //reduce0
    blocks=(blocks-1)/(2*th)+1;  //reduce3
    cudaDeviceSynchronize();
  }
  reduce3_i<<<blocks,th,shared_mem_size>>>(*in,*out,n);
  cudaMemcpy(&result,*out,sizeof(int),cudaMemcpyDeviceToHost);

  cudaFree(d1_out);
  cudaFree(d2_out);
  d1_out=NULL;
  d2_out=NULL;

  return result;
}


float volume_dev(float*d_u,int cnum){
  float result;

  int n=CIMAX*CJMAX*cnum;
  int th=WARP;
  int blocks=(n-1)/th+1;

  int shared_mem_size=2*th*sizeof(float);

  float *d1_out, *d2_out;
  cudaMalloc((void**)&d1_out,sizeof(float)*blocks);
  cudaMalloc((void**)&d2_out,sizeof(float)*blocks);

  float **in=&d_u,**out=&d1_out;
  while (blocks>1) {
    reduce0<<<blocks, th, shared_mem_size>>>(*in, *out, n);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
      puts(cudaGetErrorString(err));
    }
    if(*out==d1_out){
      out=&d2_out; in=&d1_out;
    }
    else{
      out=&d1_out; in=&d2_out;
    }
    n=blocks;
    blocks=(blocks-1)/th+1;
    cudaDeviceSynchronize();
  }
  reduce0<<<blocks,th,shared_mem_size>>>(*in,*out,n);
  cudaMemcpy(&result,*out,sizeof(float),cudaMemcpyDeviceToHost);

  cudaFree(d1_out);
  cudaFree(d2_out);
  d1_out=NULL;
  d2_out=NULL;

  return result*DX*DY;
}


// void CoM(float**d_u,float**d_u_temp,int* d){

//   dim3 cgrids;
//   dim3 cblocks;
//   cgrids.x = CIMAX*CJMAX/WARP;
//   cgrids.y = 1;
//   cgrids.z = 1;
//   cblocks.x = WARP;
//   cblocks.y = 1;
//   cblocks.z = 1;

//   for(int n=0;n<NMAX;n++){ h_ijk[n].di=0; h_ijk[n].dj=0; }

//   float *d_ux,*d_uy;
//   cudaMalloc((void**)&d_ux,sizeof(float)*CIMAX*CJMAX);
//   cudaMalloc((void**)&d_uy,sizeof(float)*CIMAX*CJMAX);
//   for(int n=0;n<1;n++){
//     sumxy_Kernel<<<cgrids,cblocks>>>(d_u,d_ux,d_uy,n);
//     cudaDeviceSynchronize();
//     float di=volume(d_ux)/h_cells[n].v-CIMAX*0.5; //i
//     float dj=volume(d_uy)/h_cells[n].v-CJMAX*0.5; //j
//     h_ijk[n].di=(int)di; h_ijk[n].dj=(int)dj;
//     //d[n*2+0]=ii; d[n*2+1]=jj;
//   }
//   cudaMemcpyToSymbol(d_ijk,&h_ijk,sizeof(PARAM::com)*NMAX);

//   CoM_Kernel<<<cgrids,cblocks>>>(d_u,d_u_temp);
//   cudaDeviceSynchronize();
//   update_Kernel<<<cgrids,cblocks>>>(d_u,d_u_temp);
//   cudaDeviceSynchronize();

//   cudaFree(d_ux); cudaFree(d_uy);
//   d_ux=NULL; d_uy=NULL;
// }

// void CoM(float**d_u,float**d_u_temp,float**d_p,float**d_p_temp,int* d,int cnum){

//   dim3 cgrids;
//   dim3 cblocks;
//   cgrids.x = CIMAX*CJMAX/WARP;
//   cgrids.y = 1;
//   cgrids.z = 1;
//   cblocks.x = WARP;
//   cblocks.y = 1;
//   cblocks.z = 1;

//   for(int n=0;n<NMAX;n++){ h_ijk[n].di=0; h_ijk[n].dj=0; }

//   float *d_ux,*d_uy;
//   cudaMalloc((void**)&d_ux,sizeof(float)*CIMAX*CJMAX);
//   cudaMalloc((void**)&d_uy,sizeof(float)*CIMAX*CJMAX);
//   for(int n=0;n<cnum;n++){
//     sumxy_Kernel<<<cgrids,cblocks>>>(d_u,d_ux,d_uy,n);
//     cudaDeviceSynchronize();
//     float di=volume(d_ux)/h_cells[n].v-CIMAX*0.5; //i
//     float dj=volume(d_uy)/h_cells[n].v-CJMAX*0.5; //j
//     h_ijk[n].di=(int)di; h_ijk[n].dj=(int)dj;
//     //d[n*2+0]=ii; d[n*2+1]=jj;
//   }
//   cudaMemcpyToSymbol(d_ijk,&h_ijk,sizeof(PARAM::com)*NMAX);

//   //CoM_Kernel<<<cgrids,cblocks>>>(d_u,d_u_temp);
//   CoM_up_Kernel<<<cgrids,cblocks>>>(d_u,d_u_temp,d_p,d_p_temp);
//   cudaDeviceSynchronize();
//   //update_Kernel<<<cgrids,cblocks>>>(d_u,d_u_temp,CIMAX);
//   update_up_Kernel<<<cgrids,cblocks>>>(d_u,d_u_temp,d_p,d_p_temp);
//   cudaDeviceSynchronize();

//   cudaFree(d_ux); cudaFree(d_uy);
//   d_ux=NULL; d_uy=NULL;
// }


void input_init_cells(float *r0,int cnum){
  string str=to_string(cnum);
  string fname="src/init_cells_"+str; 
  ifstream fin(fname); 
  if(!fin.is_open()){cerr<<"ERROR:Could not open input file: init_cells"<<endl;exit(8);}
  int a=0;
  fin>>a;
  cout<<cnum<<endl;
  //#pragma omp parallel for
  for(int n=0;n<cnum;n++){
    h_cells[n].Gx=0.0f; h_cells[n].Gy=0.0f;
    fin>>r0[n]>>h_cells[n].Gx>>h_cells[n].Gy;
    //cout<<r0[n]<<" "<<h_cells[n].Gx<<" "<<h_cells[n].Gy<<endl;
    h_cells[n].cimin=h_cells[n].Gx/DX-CIMAX*0.50f;
    h_cells[n].cjmin=h_cells[n].Gy/DY-CJMAX*0.50f;
    cout<<h_cells[n].cimin<<" "<<h_cells[n].cjmin<<endl;
  }
  fin.close();
}


void time_evolution_poles(float* m_u,float* e_eta,float (&r1)[2],float (&r2)[2],int cnum,int n){

  srand((unsigned int)time(NULL));

  dim3 grids;
  dim3 blocks;
  grids.x = CIMAX*CJMAX/WARP;
  grids.y = 1;
  grids.z = 1;
  blocks.x = WARP;
  blocks.y = 1;
  blocks.z = 1;

  float r1b[2]={}; float r2b[2]={};
  float *f1x,*f1y,*f2x,*f2y;
  cudaMalloc((float**)&f1x,sizeof(float)*CIMAX*CJMAX);
  cudaMalloc((float**)&f1y,sizeof(float)*CIMAX*CJMAX);
  cudaMalloc((float**)&f2x,sizeof(float)*CIMAX*CJMAX);
  cudaMalloc((float**)&f2y,sizeof(float)*CIMAX*CJMAX);
  float F1[2]={}; float F2[2]={};

  //noise
  float theta;
  float noise1[2]={}; float noise2[2]={};

  int CFGcount=0;
  int CFGcount2=0;
  while(CFGcount<5000 && CFGcount2<50000){
    //cout<<"CFG "<<CFGcount<<endl;
    for(int d=0;d<2;d++){r1b[d]=r1[d]; r2b[d]=r2[d]; F1[d]=0; F2[d]=0;}

    force_field_Kernel<<<grids,blocks>>>(m_u,e_eta,r1[0],r1[1],r2[0],r2[1],f1x,f1y,f2x,f2y,n);
    F1[0]=volume(f1x); F1[1]=volume(f1y);
    F2[0]=volume(f2x); F2[1]=volume(f2y);
    //cout<<F1[0]<<" "<<F1[1]<<" "<<F2[0]<<" "<<F2[1]<<endl;

    //noise
    if(cnum!=1){
      theta=rand()*M_PI*2.0f;
      noise1[0]=-cosf(theta)*0.001f; noise1[1]=-sinf(theta)*0.001f;
      noise2[0]=cosf(theta+M_PI)*0.001f; noise2[1]=sinf(theta+M_PI)*0.001f;
    }

    for(int d=0;d<2;d++){
      if(CFGcount>4000 || CFGcount2>40000){
 	r1[d]+=(F1[d]-h_para.sigma*(distance_h(r1,r2)-h_para.ls)
		*((r1[d]-r2[d])/distance_h(r1,r2)))/h_para.mu*h_sys.dts;
 	r2[d]+=(F2[d]-h_para.sigma*(distance_h(r1,r2)-h_para.ls)
		*((r2[d]-r1[d])/distance_h(r1,r2)))/h_para.mu*h_sys.dts;
      }
      else{
 	// add noise 20191127----------------------- start
 	r1[d]+=(F1[d]-h_para.sigma*(distance_h(r1,r2)-h_para.ls)
		*((r1[d]-r2[d])/distance_h(r1,r2)))/h_para.mu*h_sys.dts+noise1[d];
 	r2[d]+=(F2[d]-h_para.sigma*(distance_h(r1,r2)-h_para.ls)
		*((r2[d]-r1[d])/distance_h(r1,r2)))/h_para.mu*h_sys.dts+noise2[d];
 	// add noise 20191127----------------------- end
      }
    }
    //cout<<r1[0]<<" "<<r1[1]<<" "<<r2[0]<<" "<<r2[1]<<endl;

    if(fabsf(r1b[0]-r1[0])<0.000001 && fabsf(r1b[1]-r1[1])<0.000001 &&
       fabsf(r2b[0]-r2[0])<0.000001 && fabsf(r2b[1]-r2[1])<0.000001){CFGcount++;}
    CFGcount2++;
  }

  if(CFGcount>4999) cout<<"division OK"<<endl;
  else if(CFGcount2>49999) cout<<"division error"<<endl;

  cudaFree(f1x); cudaFree(f1y);
  cudaFree(f2x); cudaFree(f2y);
  f1x=NULL; f1y=NULL;
  f2x=NULL; f2y=NULL;
}

void division(float*u,float(&r1)[2],float(&r2)[2],int m,int n){

  dim3 grids;
  dim3 blocks;
  grids.x = CIMAX*CJMAX/WARP;
  grids.y = 1;
  grids.z = 1;
  blocks.x = WARP;
  blocks.y = 1;
  blocks.z = 1;

  float *m_u_temp;
  cudaMalloc((float**)&m_u_temp,sizeof(float)*CIMAX*CJMAX);
  copy_mother_cell_Kernel<<<grids,blocks>>>(u,m_u_temp,m);
  cudaDeviceSynchronize();

  cout<<"set daughter cells...";
  set_daughter_cells_Kernel<<<grids,blocks>>>(m_u_temp,u,r1[0],r1[1],r2[0],r2[1],m,n);
  cudaDeviceSynchronize();
  cout<<"Done."<<endl;

  cudaFree(m_u_temp);
  m_u_temp=NULL;
}

void output_all_usc(int t,string Dir,float*u,float*s,float*c){

  string fnameu,fnames,fnamec;
  fnameu=Dir+"/u_"+to_string(t)+".dat";
  fnames=Dir+"/s_"+to_string(t)+".dat";
  fnamec=Dir+"/c_"+to_string(t)+".dat";

  ofstream foutu(fnameu); ofstream fouts(fnames); ofstream foutc(fnamec); 
  if(!foutu.is_open()){cerr<<"ERROR:Could not open output file u"<<endl;exit(8);}
  if(!fouts.is_open()){cerr<<"ERROR:Could not open output file s"<<endl;exit(8);}
  if(!foutc.is_open()){cerr<<"ERROR:Could not open output file c"<<endl;exit(8);}
  for(int j=JMAX-2;j>=0;j=j-2){
    for(int i=0;i<IMAX;i=i+2){
      foutu<<u[i+IMAX*j]<<" ";fouts<<s[i+IMAX*j]<<" ";foutc<<c[i+IMAX*j]<<" ";
    }
    foutu<<endl; fouts<<endl; foutc<<endl;
  }
  foutu.close(); fouts.close(); foutc.close();
}



void e_eta(float*d_e_eta,float* hd_u_m1,float*d_u,int m1,int cnum){

  string fname;
  dim3 grids;
  dim3 blocks;
  grids.x = IMAX*JMAX/WARP;
  grids.y = 1;
  grids.z = 1;
  blocks.x = WARP;
  blocks.y = 1;
  blocks.z = 1;

  float *d_u_m1;
  cudaMalloc((float**)&d_u_m1,sizeof(float)*IMAX*JMAX);
  init_all_Kernel<<<grids,blocks>>>(d_u_m1);
  cudaDeviceSynchronize();
  grids.x = CIMAX*CJMAX/WARP;
  u_m1_Kernel<<<grids,blocks>>>(d_u_m1,hd_u_m1,m1);
  cudaDeviceSynchronize();

  for(int m2=0;m2<cnum;m2++){
    if(m2!=m1){
      e_eta_Kernel<<<grids,blocks>>>(d_e_eta,d_u,d_u_m1,m2);
      cudaDeviceSynchronize();
    }
  }
  cudaFree(d_u_m1);
  d_u_m1=NULL;
}

void update_lumen_pressure(float *d_u_all,float *d_s,float volume_s){
  dim3 grids;
  dim3 blocks;
  grids.x = IMAX*JMAX/WARP;
  grids.y = 1;
  grids.z = 1;
  blocks.x = WARP;
  blocks.y = 1;
  blocks.z = 1;

  //feedback control of lumen pressure 2
  float *d_overlap;
  cudaMalloc((float**)&d_overlap,sizeof(float)*IMAX*JMAX);
  init_all_Kernel<<<grids,blocks>>>(d_overlap);
  cudaDeviceSynchronize();
  overlap_us_Kernel<<<grids,blocks>>>(d_u_all,d_s,d_overlap);
  cudaDeviceSynchronize();
  float sum_overlap=volume_all(d_overlap);

  if(sum_overlap>0.0f) h_xi=h_para.xi*sum_overlap/volume_s;
  else h_xi=0.0f;

  cudaMemcpyToSymbol(d_xi,&h_xi,sizeof(float));
  cudaFree(d_overlap);
  d_overlap=NULL;
}


void timestamp(const int time,string Dir){
  string fname;
  fname=Dir+"/timestamp.dat";
  ofstream fout(fname.c_str()); 
  if(!fout.is_open()){cerr<<"ERROR:Could not open output: timestamp"<<fname<<endl;exit(8);}
  fout<<time<<endl;
  fout.close();
}

float h_h(float u){
  return u*u*(3.0f-2.0f*u);
}


/////////////////////////////////////////////////////////////////////////////////
//
//  main code
//
/////////////////////////////////////////////////////////////////////////////////


int main(int argc, char *argv[]){

  string fname;

  //Mersenne twister
  //my_srand(atoi(argv[1]));
  my_srand(34);

  // input parameters--------------------------------------------------start

  string Dir="DATA/"+string(argv[1]);

  int tmax   = std::round(T/DT);
  int out_dt = std::round(TOUT/DT);
  int out_dt2= std::round(TOUT2/DT);
  cout<<"tmax="<<tmax<<" out_dt="<<out_dt<<" out_dt2="<<out_dt2<<endl;

  float T_th = -1.0f;

  h_sys.dx=DX; h_sys.dy=DY;
  h_sys.imax=IMAX; h_sys.jmax=JMAX;
  h_sys.cimax=CIMAX; h_sys.cjmax=CJMAX;
  h_sys.dt =DT;
  h_sys.nmax=NMAX;
  h_sys.dts=DTS;
  cudaMemcpyToSymbol(d_sys,&h_sys,sizeof(PARAM::psys));

  h_para.D_u    =0.001f;
  h_para.tau_u  =1.0f;
  h_para.V      =3.0f;
  h_para.vd     =0.1f;
  h_para.alpha  =1.0f;
  h_para.beta   =1.0f;
  h_para.eta    =0.0075f;
  h_para.gamma  =0.0f;
  h_para.gamma_curv=0.015f;

  float tau_V   =atof(argv[2]);
  float noise_tau_V  =0.25f;
  h_para.alpha_V=1.00f;

  h_para.D_s    =0.001f;
  h_para.beta_s =1.0f;
  h_para.eta_s  =0.000f;
  h_para.gamma_s=0.00f;
  h_para.tau_s  =1.0f;
  h_para.xi     =atof(argv[3]);
  float xi2     =0.005f;
  h_para.alpha_s=1.0f;
  h_para.p_st   =0.8f;
  h_para.v_t    =0.35f;

  h_para.D_c    =0.001f;
  h_para.beta_cu=1.0f;
  h_para.beta_cs=1.0f;
  h_para.eta_cu =0.001f;
  h_para.gamma_c=0.00f;
  h_para.tau_c  =1.0f;
  h_para.xi_c   =0.005f;
  h_para.alpha_c=0.001f;

  h_para.ep_d   =0.100f;
  h_para.rho0   =0.010f;
  h_para.rhoe   =5.000f;
  h_para.mu     =1.000f;
  h_para.sigma  =0.001f;
  h_para.ls     =0.000f;

  h_para.D_p    =0.001f;
  h_para.tau_p  =1.0f;
  h_para.alpha_p=1.0f;
  h_para.eta_ps =0.000f;
  h_para.gamma_p=0.00f;
  h_para.Vp     =0.01f;
  h_para.C_p    =0.7f;
  h_para.w_p    =1.0f;
  h_para.p_r    =0.50f;
  h_para.k_p    =1.0f;
  h_para.p_th   =0.8f;
  float p_th2   =1.0f;
  h_para.l_anti =0.02f;

  cudaMemcpyToSymbol(d_para,&h_para,sizeof(PARAM::param));

  float tauV[NMAX]={};
  for(int n=0;n<NMAX;n++){
    h_cells[n].Gx=0.0f; h_cells[n].Gy=0.0f;
    h_cells[n].cimin=0; h_cells[n].cjmin=0;
    h_cells[n].v=0.0f; h_cells[n].targetv=0.0f;
    h_cells[n].vp=0.0f;
    tauV[n]=0.0f;
  }
  float volume_s=0.0f;
  float volume_c=0.0f;

  //int cnum=atoi(argv[2]);
  int cnum=8;
  cudaMemcpyToSymbol(d_cnum,&cnum,sizeof(int));

  float *h_r0;
  h_r0=(float*)malloc(sizeof(float)*cnum);
  for(int i=0;i<cnum;i++) h_r0[i]=0.0f;
  input_init_cells(h_r0,cnum);

  for(int n=0;n<NMAX;n++){
    h_ijk[n].di=0; h_ijk[n].dj=0;
  }

  float init_theta=0.0;

  //reshaping
  float dtau=0.04f;//<=dx*dx/sqrt(D)/4 (Olsson 2005)
  float tol=0.001f;
  // input parameters--------------------------------------------------end

  // definision--------------------------------------------------start
  dim3 cgrids(CIMAX*CJMAX/WARP,1,1), cblocks(WARP,1,1);
  dim3 ccgrids(CIMAX*CJMAX*NMAX/WARP,1,1), ccblocks(WARP,1,1);
  dim3 grids(IMAX*JMAX/WARP,1,1), blocks(WARP,1,1);

  //cells
  float *h_u,*d_u,*d_u_temp;
  h_u = (float*)malloc(sizeof(float)*NMAX*CIMAX*CJMAX);
  cudaMalloc((void **)&d_u,sizeof(float)*NMAX*CIMAX*CJMAX);
  cudaMalloc((void **)&d_u_temp,sizeof(float)*NMAX*CIMAX*CJMAX);
  for(int i=0;i<NMAX*CIMAX*CJMAX;i++) h_u[i]=0.0f;
  cudaMemcpy(d_u,h_u,sizeof(float)*NMAX*CIMAX*CJMAX,cudaMemcpyHostToDevice);
  cudaMemcpy(d_u_temp,h_u,sizeof(float)*NMAX*CIMAX*CJMAX,cudaMemcpyHostToDevice);

  //anti-adhesive molecules
  float *h_p,*d_p,*d_p_temp;
  h_p = (float*)malloc(sizeof(float)*NMAX*CIMAX*CJMAX);
  cudaMalloc((void **)&d_p,sizeof(float)*NMAX*CIMAX*CJMAX);
  cudaMalloc((void **)&d_p_temp,sizeof(float)*NMAX*CIMAX*CJMAX);
  for(int i=0;i<NMAX*CIMAX*CJMAX;i++) h_p[i]=0.0f; 
  cudaMemcpy(d_p,h_p,sizeof(float)*NMAX*CIMAX*CJMAX,cudaMemcpyHostToDevice);
  cudaMemcpy(d_p_temp,h_p,sizeof(float)*NMAX*CIMAX*CJMAX,cudaMemcpyHostToDevice);

  //lumen
  float *h_s,*d_s,*d_s_temp;
  h_s = (float *)malloc(sizeof(float)*IMAX*JMAX);
  cudaMalloc((void **)&d_s,sizeof(float)*IMAX*JMAX);
  cudaMalloc((void **)&d_s_temp,sizeof(float)*IMAX*JMAX);
  for(int i=0;i<IMAX*JMAX;i++) h_s[i] = 0; 
  cudaMemcpy(d_s,h_s,sizeof(float)*IMAX*JMAX,cudaMemcpyHostToDevice);
  cudaMemcpy(d_s_temp,h_s,sizeof(float)*IMAX*JMAX,cudaMemcpyHostToDevice);

  //medium
  float *h_c,*d_c,*d_c_temp;
  h_c = (float *)malloc(sizeof(float)*IMAX*JMAX);
  cudaMalloc((void **)&d_c,sizeof(float)*IMAX*JMAX);
  cudaMalloc((void **)&d_c_temp,sizeof(float)*IMAX*JMAX);
  for(int i=0;i<IMAX*JMAX;i++) h_c[i] = 0; 
  cudaMemcpy(d_c,h_c,sizeof(float)*IMAX*JMAX,cudaMemcpyHostToDevice);
  cudaMemcpy(d_c_temp,h_c,sizeof(float)*IMAX*JMAX,cudaMemcpyHostToDevice);

  //reshaping
  float *d_dev_u,*d_vx_u,*d_vy_u;
  cudaMalloc((void **)&d_dev_u,sizeof(float)*NMAX*CIMAX*CJMAX);
  cudaMalloc((void **)&d_vx_u,sizeof(float)*NMAX*CIMAX*CJMAX);
  cudaMalloc((void **)&d_vy_u,sizeof(float)*NMAX*CIMAX*CJMAX);

  float *d_dev_s,*d_vx_s,*d_vy_s;
  cudaMalloc((void **)&d_dev_s,sizeof(float)*IMAX*JMAX);
  cudaMalloc((void **)&d_vx_s,sizeof(float)*IMAX*JMAX);
  cudaMalloc((void **)&d_vy_s,sizeof(float)*IMAX*JMAX);

  float *d_dev_c,*d_vx_c,*d_vy_c;
  cudaMalloc((void **)&d_dev_c,sizeof(float)*IMAX*JMAX);
  cudaMalloc((void **)&d_vx_c,sizeof(float)*IMAX*JMAX);
  cudaMalloc((void **)&d_vy_c,sizeof(float)*IMAX*JMAX);

  //\sum_m u_m & \sum_m p_m
  float *h_uall,*d_uall;
  h_uall = (float *)malloc(sizeof(float)*IMAX*JMAX);
  cudaMalloc((float**)&d_uall,sizeof(float)*IMAX*JMAX);
  float *h_pall,*d_pall;
  h_pall = (float *)malloc(sizeof(float)*IMAX*JMAX);
  cudaMalloc((float**)&d_pall,sizeof(float)*IMAX*JMAX);
  for(int i=0;i<IMAX*JMAX;i++){ h_uall[i] = 0; h_pall[i] = 0; }
  cudaMemcpy(d_uall,h_uall,sizeof(float)*IMAX*JMAX,cudaMemcpyHostToDevice);
  cudaMemcpy(d_pall,h_pall,sizeof(float)*IMAX*JMAX,cudaMemcpyHostToDevice);

  //each u & p
  float *d_u_n, *d_p_n;
  cudaMalloc((float**)&d_u_n,sizeof(float)*CIMAX*CJMAX);
  cudaMalloc((float**)&d_p_n,sizeof(float)*CIMAX*CJMAX);

  //\sum_m h(u_m)
  float *d_phi;
  cudaMalloc((void **)&d_phi,sizeof(float)*IMAX*JMAX);

  //CFG
  float *d_e_eta;
  cudaMalloc((float**)&d_e_eta,sizeof(float)*IMAX*JMAX);

  //anti-adhesion
  float *d_u_adhe;
  cudaMalloc((float**)&d_u_adhe,sizeof(float)*IMAX*JMAX);

  //CoM
  float *d_ux,*d_uy;
  cudaMalloc((void**)&d_ux,sizeof(float)*CIMAX*CJMAX);
  cudaMalloc((void**)&d_uy,sizeof(float)*CIMAX*CJMAX);
  // definision--------------------------------------------------end


  // initial condition--------------------------------------------------start
  ccgrids.x = CIMAX*CJMAX*cnum/WARP;
  float *d_r0;
  cudaMalloc((void **)&d_r0,sizeof(float)*cnum);
  cudaMemcpy(d_r0,h_r0,sizeof(float)*cnum,cudaMemcpyHostToDevice);

  init_u_Kernel<<<ccgrids,ccblocks>>>(d_u,d_r0);
  cudaDeviceSynchronize();
  cudaFree(d_r0); d_r0=NULL;


  //cell & p volume
  cgrids.y = 1;
  for(int n=0;n<cnum;n++){
    u_n_Kernel<<<cgrids,cblocks>>>(d_u_n,d_u,n);
    cudaDeviceSynchronize();
    h_cells[n].v=volume_h(d_u_n);

    if(h_para.alpha_V>0.0f){
      h_dtv[n].dtargetv=h_cells[n].v*1.10;
      h_cells[n].targetv=(float)h_dtv[n].dtargetv;
    }
    else h_cells[n].targetv=h_para.V;
  }
  cudaMemcpyToSymbol(d_cells,&h_cells,sizeof(PARAM::cells)*NMAX);

  //target cell volume
  for(int n=0;n<cnum;n++) tauV[n]=tau_V+(MT_rand()*2.0f-1.0f)*noise_tau_V*tau_V;

  // uall&pall&phi
  init_all_uphi_Kernel<<<grids,blocks>>>(d_uall,d_phi);
  cudaDeviceSynchronize();
  cgrids.y = 1;
  for(int n=0;n<cnum;n++){
    all_uphi_Kernel<<<cgrids,cblocks>>>(d_uall,d_phi,d_u,n);
    cudaDeviceSynchronize();
  }

  init_all_Kernel<<<grids,blocks>>>(d_u_adhe);	  
  cudaDeviceSynchronize();
  for(int m=0;m<cnum;m++){
    float *d_u_m;
    cudaMalloc((float**)&d_u_m,sizeof(float)*CIMAX*CJMAX);
    u_n_Kernel<<<cgrids,cblocks>>>(d_u_m,d_u,m);
    cudaDeviceSynchronize();
    e_eta(d_u_adhe,d_u_m,d_u,m,cnum);
    cudaFree(d_u_m);
    d_u_m=NULL;
  }


  // medium
  init_medium_Kernel<<<grids,blocks>>>(d_c,d_uall);
  cudaDeviceSynchronize();
  grids.x = IMAX/WARP;
  boundary_dirichlet_Kernel<<<grids,blocks>>>(d_c,d_c_temp);
  cudaDeviceSynchronize();
  volume_c=volume_all(d_c);
  grids.x = IMAX*JMAX/WARP;

  //calculation area for lumen & medium
  int imin_sc,jmin_sc,imax_sc,jmax_sc,imin_temp,jmin_temp;
  dim3 grids_sc, blocks_sc;
  imin_sc=h_cells[0].cimin; jmin_sc=h_cells[0].cjmin;
  imin_temp=h_cells[0].cimin; jmin_temp=h_cells[0].cjmin;
  for(int n=1;n<cnum;n++){
    if(imin_sc>h_cells[n].cimin) imin_sc=h_cells[n].cimin;
    if(jmin_sc>h_cells[n].cjmin) jmin_sc=h_cells[n].cjmin;
    if(imin_temp<h_cells[n].cimin) imin_temp=h_cells[n].cimin;
    if(jmin_temp<h_cells[n].cjmin) jmin_temp=h_cells[n].cjmin;
    }
  imax_sc=(imin_temp-imin_sc)+CIMAX;
  if(imax_sc>IMAX) imax_sc=IMAX;
  jmax_sc=(jmin_temp-jmin_sc)+CJMAX;
  if(jmax_sc>JMAX) jmax_sc=JMAX;

  cout<<imin_sc<<" "<<jmin_sc<<" "<<imax_sc<<" "<<jmax_sc<<endl;
  grids_sc.x=(imax_sc*jmax_sc-1)/WARP+1;
  blocks_sc.x=WARP;
  // initial condition--------------------------------------------------end


  // output initial condition--------------------------------------------------start
  cudaMemcpy(h_uall, d_uall, sizeof(float)*IMAX*JMAX, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c, d_c, sizeof(float)*IMAX*JMAX, cudaMemcpyDeviceToHost);
  output_all_usc(0,Dir,h_uall,h_s,h_c);

  // cell volume
  fname=Dir+"/Volume.dat";
  ofstream fout_vol(fname.c_str()); 
  if(!fout_vol.is_open()){cerr<<"ERROR:Could not open output: Volume"<<endl;exit(8);}
  fout_vol<<"0 "<<volume_s<<" "<<volume_c<<" ";
  for(int n=0;n<NMAX;n++) fout_vol<<h_cells[n].v<<" "
				  <<h_dtv[n].dtargetv<<" "
				  <<h_cells[n].vp<<" ";
  fout_vol<<endl;

  //output file number of cell & center
  fname=Dir+"/ncell_t.dat";
  ofstream fout_nc(fname.c_str());
  if(!fout_nc.is_open()){cerr<<"ERROR:Could not open output: number of cell data"<<endl;exit(8);}
  fout_nc<<"0 "<<cnum<<" ";
  for(int n=0;n<NMAX;n++) fout_nc<<fixed<<setprecision(6)<<h_cells[n].Gx<<" "<<h_cells[n].Gy<<" ";
  fout_nc<<endl;
  // output initial condition--------------------------------------------------end


  // time evolution--------------------------------------------------start
  cout<<"calculation start..."<<endl;

  int tstop=0;
  for(int t=1;t+tstop<tmax+1;t++){

    if(cnum==NMAX&&h_para.alpha_s>0.1f){
      float ave_v=0.0f;
      for(int n=0;n<NMAX;n++) ave_v+=h_cells[n].v;
      ave_v=ave_v/NMAX;
      if(ave_v>2.96f){
        h_para.alpha_s=0.0f;
        cudaMemcpyToSymbol(d_para,&h_para,sizeof(PARAM::param));
	cout<<"alpha_s=0.0"<<endl;
      }
    }
    if(t==(int)(T_th/DT)){
      h_para.xi=xi2;
      h_para.p_th=p_th2*h_para.p_th;
      cudaMemcpyToSymbol(d_para,&h_para,sizeof(PARAM::param));
      cout<<"pressure decrease t="<<T_th<<endl;
    }

    // time evolution--------------------------------------------------start
    cgrids.y = cnum;
    ccgrids.x = CIMAX*CJMAX*cnum/WARP;
    time_evolution_u_with_reshaping_Kernel<<<ccgrids,ccblocks>>>(d_u_temp,d_u,
     								  d_phi,d_s,d_c);

    time_evolution_sc_with_reshaping_2_Kernel<<<grids_sc,blocks_sc>>>(d_c_temp,d_c,d_s_temp,d_s,
								      d_phi,d_uall,
								      imin_sc,jmin_sc,imax_sc,jmax_sc,
								      volume_c);
    cudaDeviceSynchronize();
    // time evolution--------------------------------------------------end

    // update--------------------------------------------------start
    update_Kernel<<<ccgrids,ccblocks>>>(d_u,d_u_temp);
    update_sc_2_Kernel<<<grids_sc,blocks_sc>>>(d_s,d_s_temp,d_c,d_c_temp,
					       imin_sc,jmin_sc,imax_sc,jmax_sc);
    cudaDeviceSynchronize();

    //volume of each cell & p
    cgrids.y = 1;
    for(int n=0;n<cnum;n++){
      up_n_Kernel<<<cgrids,cblocks>>>(d_u_n,d_u,d_p_n,d_p,n);
      cudaDeviceSynchronize();
      h_cells[n].v=volume_h(d_u_n);

      h_dtv[n].dtargetv
	+=(double)(h_sys.dt/tauV[n]*h_para.alpha_V)*((double)h_para.V-h_dtv[n].dtargetv);

      h_cells[n].targetv=(float)h_dtv[n].dtargetv;

    }
    cudaMemcpyToSymbol(d_cells,&h_cells,sizeof(PARAM::cells)*NMAX);


    //CoM
    if(t%out_dt==0){
      cgrids.y = 1;
      for(int n=0;n<cnum;n++){
	sumxy_Kernel<<<cgrids,cblocks>>>(d_u,d_ux,d_uy,n);
	cudaDeviceSynchronize();
	float di=volume(d_ux)/h_cells[n].v-CIMAX*0.5f; //i
	float dj=volume(d_uy)/h_cells[n].v-CJMAX*0.5f; //j
	h_ijk[n].di=(int)di; h_ijk[n].dj=(int)dj;
	h_cells[n].cimin+=(int)di; h_cells[n].cjmin+=(int)dj;
	h_cells[n].Gx+=(float)(((int)di)*DX);
	h_cells[n].Gy+=(float)(((int)dj)*DY);
      }
      cudaMemcpyToSymbol(d_ijk,&h_ijk,sizeof(PARAM::com)*NMAX);
      cudaMemcpyToSymbol(d_cells,&h_cells,sizeof(PARAM::cells)*NMAX);

      cgrids.y = cnum;
      CoM_up_Kernel<<<cgrids,cblocks>>>(d_u,d_u_temp,d_p,d_p_temp);
      cudaDeviceSynchronize();
      update_Kernel<<<ccgrids,ccblocks>>>(d_u,d_u_temp);
      cudaDeviceSynchronize();


      //calculation area for lumen & medium
      imin_sc=h_cells[0].cimin; jmin_sc=h_cells[0].cjmin;
      imin_temp=h_cells[0].cimin; jmin_temp=h_cells[0].cjmin;
      for(int n=1;n<cnum;n++){
	if(imin_sc>h_cells[n].cimin) imin_sc=h_cells[n].cimin;
	if(jmin_sc>h_cells[n].cjmin) jmin_sc=h_cells[n].cjmin;
	if(imin_temp<h_cells[n].cimin) imin_temp=h_cells[n].cimin;
	if(jmin_temp<h_cells[n].cjmin) jmin_temp=h_cells[n].cjmin;
      }
      imax_sc=(imin_temp-imin_sc)+CIMAX;
      if(imax_sc>IMAX) imax_sc=IMAX;
      jmax_sc=(jmin_temp-jmin_sc)+CJMAX;
      if(jmax_sc>JMAX) jmax_sc=JMAX;

      grids_sc.x=(imax_sc*jmax_sc-1)/WARP+1;
      blocks_sc.x=WARP;

      //cell reach to edge of system
      for(int n=0;n<cnum;n++){
	if(h_cells[n].cimin<1 || h_cells[n].cimin+CIMAX>IMAX-1 || 
	   h_cells[n].cjmin<1 || h_cells[n].cjmin+CJMAX>JMAX-1){
	   tstop=tmax;
	   cout<<"a cell reaches to edge of system."<<endl;
	   }
      }

    }
    // update--------------------------------------------------end


    // intermediate step--------------------------------------------------start
    cgrids.x = CIMAX*CJMAX/WARP;
    cgrids.y = cnum;
    cblocks.x = WARP;
    normal_vector_field_Kernel<<<ccgrids,ccblocks>>>(d_vx_u,d_vy_u,d_u);
    cudaDeviceSynchronize();

    cgrids.x = CIMAX/32;
    cblocks.x = 32;
    boundary_normal_vector_field_Kernel<<<cgrids,cblocks>>>(d_vx_u,d_vy_u);
    cudaDeviceSynchronize();

    cgrids.x = CIMAX*CJMAX/WARP;
    cblocks.x = WARP;
    int count=0;
    float dev=cnum*tol*dtau*2.0f;
    while(dev>cnum*tol*dtau){
      reshaping_Kernel<<<ccgrids,ccblocks>>>(d_vx_u,d_vy_u,d_u,d_u_temp,dtau);
      cudaDeviceSynchronize();

      dev_Kernel<<<cgrids,cblocks>>>(d_dev_u,d_u,d_u_temp);
      cudaDeviceSynchronize();

      dev=0.0f;
      for(int n=0;n<cnum;n++){
     	u_n_Kernel<<<cgrids,cblocks>>>(d_u_n,d_dev_u,n);
     	cudaDeviceSynchronize();
     	dev+=volume(d_u_n);
      }

      cgrids.y = cnum;
      update_Kernel<<<ccgrids,ccblocks>>>(d_u,d_u_temp);
      cudaDeviceSynchronize();
      count++;
    }

    // reshaping s & c
    if(volume_s>0.0f){
      normal_vector_field_sc_2_Kernel<<<grids_sc,blocks_sc>>>(d_vx_s,d_vy_s,d_s,d_vx_c,d_vy_c,d_c,
							      imin_sc,jmin_sc,imax_sc,jmax_sc);
      cudaDeviceSynchronize();
      grids.x = IMAX/WARP;
      if(imin_sc==0 || jmin_sc==0 || imax_sc==IMAX-1 || jmax_sc==JMAX-1)
	boundary_normal_vector_field_sc_Kernel<<<grids,blocks>>>(d_vx_s,d_vy_s,d_vx_c,d_vy_c);
      cudaDeviceSynchronize();
    }
    else{
      normal_vector_field_all_2_Kernel<<<grids_sc,blocks_sc>>>(d_vx_c,d_vy_c,d_c,imin_sc,jmin_sc,imax_sc,jmax_sc);
      cudaDeviceSynchronize();
      grids.x = IMAX/WARP;
      if(imin_sc==0 || jmin_sc==0 || imax_sc==IMAX-1 || jmax_sc==JMAX-1)
	boundary_normal_vector_field_all_Kernel<<<grids,blocks>>>(d_vx_c,d_vy_c);
      cudaDeviceSynchronize();
    }

    grids.x = IMAX*JMAX/WARP;
    count=0;
    dev=tol*dtau*2.0f;

    if(volume_s>0.0f){
      while(dev>tol*dtau){
	reshaping_all_2_Kernel<<<grids_sc,blocks_sc>>>(d_vx_s,d_vy_s,d_s,d_s_temp,dtau,h_para.D_s,
						       imin_sc,jmin_sc,imax_sc,jmax_sc);
	cudaDeviceSynchronize();

	dev_all_2_Kernel<<<grids_sc,blocks_sc>>>(d_dev_s,d_s,d_s_temp,imin_sc,jmin_sc,imax_sc,jmax_sc);
	cudaDeviceSynchronize();

	dev=volume_all(d_dev_s);

	update_all_2_Kernel<<<grids_sc,blocks_sc>>>(d_s,d_s_temp,imin_sc,jmin_sc,imax_sc,jmax_sc);
	cudaDeviceSynchronize();

	count++;
      }
    }

    count=0;
    dev=tol*dtau*2.0f;
    while(dev>tol*dtau){
      reshaping_all_2_Kernel<<<grids_sc,blocks_sc>>>(d_vx_c,d_vy_c,d_c,d_c_temp,dtau,h_para.D_c,
						     imin_sc,jmin_sc,imax_sc,jmax_sc);
      cudaDeviceSynchronize();

      dev_all_2_Kernel<<<grids_sc,blocks_sc>>>(d_dev_c,d_c,d_c_temp,imin_sc,jmin_sc,imax_sc,jmax_sc);
      cudaDeviceSynchronize();

      dev=volume_all(d_dev_c);

      update_all_2_Kernel<<<grids_sc,blocks_sc>>>(d_c,d_c_temp,imin_sc,jmin_sc,imax_sc,jmax_sc);
      cudaDeviceSynchronize();

      count++;
    }
    // intermediate step--------------------------------------------------end


    // cell division---------------------------------------------------start
    bool e_eta_ON=false;
    for(int n=0;n<cnum;n++){

      if(h_cells[n].v>h_para.V-h_para.vd && cnum<NMAX){
	cgrids.y = 1;
	u_n_Kernel<<<cgrids,cblocks>>>(d_u_n,d_u,n);
	cudaDeviceSynchronize();

 	if(e_eta_ON==false){
	  init_all_Kernel<<<grids,blocks>>>(d_e_eta);	  
	  for(int m=0;m<cnum;m++){
	    float *d_u_m;
	    cudaMalloc((float**)&d_u_m,sizeof(float)*CIMAX*CJMAX);
	    u_n_Kernel<<<cgrids,cblocks>>>(d_u_m,d_u,m);
	    cudaDeviceSynchronize();
	    e_eta(d_e_eta,d_u_m,d_u,m,cnum);
	    cudaFree(d_u_m);
	    d_u_m=NULL;
	  }

 	  e_eta_ON=true;
 	}

 	//cell division
	h_cells[cnum]=h_cells[n];

	//set the angle of the division plane
	float theta=0.0f;
	if(cnum==1 && init_theta>=0.0f)
	  theta=(init_theta+90.0f)/360.0f*M_PI*2.0f;
	else
	  theta=MT_rand()*M_PI*2.0f;

	float r1[2]={}; float r2[2]={};
	r1[0]=h_cells[n].Gx-cos(theta)*h_sys.dx; r1[1]=h_cells[n].Gy-sin(theta)*h_sys.dy;
	r2[0]=h_cells[n].Gx+cos(theta)*h_sys.dx; r2[1]=h_cells[n].Gy+sin(theta)*h_sys.dy;

	time_evolution_poles(d_u_n,d_e_eta,r1,r2,cnum,n);

	float Pc[2]={};
	Pc[0]=(float)(r1[0]+r2[0])*0.50f; Pc[1]=(float)(r1[1]+r2[1])*0.50f;
	set_seed_s_Kernel<<<cgrids,cblocks>>>(d_u,d_s,Pc[0],Pc[1],n);
	
	grids.y = cnum;
	set_daughter_cells_Kernel<<<grids,blocks>>>(d_u,d_u_temp,r1[0],r1[1],r2[0],r2[1],n,cnum);
	cudaDeviceSynchronize();
	cudaMemcpy(d_u,d_u_temp,sizeof(float)*NMAX*CIMAX*CJMAX,cudaMemcpyDeviceToDevice);

	//volume
	u_n_Kernel<<<cgrids,cblocks>>>(d_u_n,d_u,n);
	cudaDeviceSynchronize();
	h_cells[n].v=volume_h(d_u_n);

	u_n_Kernel<<<cgrids,cblocks>>>(d_u_n,d_u,cnum);
	cudaDeviceSynchronize();
	h_cells[cnum].v=volume_h(d_u_n);

	cout<<h_cells[n].v<<" "<<h_cells[cnum].v<<endl;
	cout<<h_cells[n].vp<<" "<<h_cells[cnum].vp<<endl;

	if(h_para.alpha_V>0.0f){
          h_dtv[cnum].dtargetv=h_cells[cnum].v*1.10f;
          h_dtv[n].dtargetv=h_cells[n].v*1.10f;
	  h_cells[cnum].targetv=(float)h_dtv[cnum].dtargetv;
	  h_cells[n].targetv=(float)h_dtv[n].dtargetv;
	}
	else{h_cells[n].targetv=h_para.V; h_cells[cnum].targetv=h_para.V;}

	cudaMemcpyToSymbol(d_cells,&h_cells,sizeof(PARAM::cells)*NMAX);

	tauV[n]=tau_V+(MT_rand()*2.0f-1.0f)*noise_tau_V*tau_V;
	tauV[cnum]=tau_V+(MT_rand()*2.0f-1.0f)*noise_tau_V*tau_V;

	cnum++;
	cudaMemcpyToSymbol(d_cnum,&cnum,sizeof(int));
      }
    }
    // cell division---------------------------------------------------end

    //uall&phi
    init_all_uphi_Kernel<<<grids,blocks>>>(d_uall,d_phi);
    cudaDeviceSynchronize();
    cgrids.y = 1;
    for(int n=0;n<cnum;n++){
      all_uphi_Kernel<<<cgrids,cblocks>>>(d_uall,d_phi,d_u,n);
      cudaDeviceSynchronize();
    }

    init_all_Kernel<<<grids,blocks>>>(d_u_adhe);	  
    cudaDeviceSynchronize();
    for(int m=0;m<cnum;m++){
      float *d_u_m;
      cudaMalloc((float**)&d_u_m,sizeof(float)*CIMAX*CJMAX);
      u_n_Kernel<<<cgrids,cblocks>>>(d_u_m,d_u,m);
      cudaDeviceSynchronize();
      e_eta(d_u_adhe,d_u_m,d_u,m,cnum);
      cudaFree(d_u_m);
      d_u_m=NULL;
    }

    volume_s=volume_all(d_s);
    volume_c=volume_all(d_c);

    if(t%out_dt==0){
      int t_real=t*DT;
      timestamp(t_real,Dir);

      //phase field
      cudaMemcpy(h_uall, d_uall, sizeof(float)*IMAX*JMAX, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_s, d_s, sizeof(float)*IMAX*JMAX, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_c, d_c, sizeof(float)*IMAX*JMAX, cudaMemcpyDeviceToHost);
      output_all_usc(t_real,Dir,h_uall,h_s,h_c);

      //volume
      fout_vol<<t_real<<" "<<volume_s<<" "<<volume_c<<" ";
      for(int n=0;n<NMAX;n++) fout_vol<<h_cells[n].v<<" "
				      <<h_dtv[n].dtargetv<<" "
				      <<h_cells[n].vp<<" ";
      fout_vol<<endl;

      //number of cells
      fout_nc<<t_real<<" "<<cnum<<" ";
      for(int n=0;n<NMAX;n++) fout_nc<<fixed<<setprecision(6)<<h_cells[n].Gx<<" "<<h_cells[n].Gy<<" ";
      fout_nc<<endl;
    }
  }
  // time evolution--------------------------------------------------end

  free(h_u); cudaFree(d_u); cudaFree(d_u_temp);
  free(h_p); cudaFree(d_p); cudaFree(d_p_temp);
  h_u=NULL; d_u=NULL; d_u_temp=NULL;
  h_p=NULL; d_p=NULL; d_p_temp=NULL;

  cudaFree(d_u_n); cudaFree(d_p_n);
  d_u_n=NULL; d_p_n=NULL;

  cudaFree(d_phi);   d_phi=NULL;
  free(h_uall); cudaFree(d_uall);
  h_uall=NULL; d_uall=NULL;
  free(h_pall); cudaFree(d_pall);
  h_pall=NULL; d_pall=NULL;

  free(h_s); cudaFree(d_s); cudaFree(d_s_temp);
  h_s=NULL; d_s=NULL; d_s_temp=NULL;

  free(h_c); cudaFree(d_c); cudaFree(d_c_temp);
  h_c=NULL; d_c=NULL; d_c_temp=NULL;

  cudaFree(d_ux); cudaFree(d_uy);
  d_ux=NULL; d_uy=NULL;

  cudaFree(d_dev_s); cudaFree(d_vx_s); cudaFree(d_vy_s);
  cudaFree(d_dev_c); cudaFree(d_vx_c); cudaFree(d_vy_c);
  d_dev_s=NULL; d_vx_s=NULL; d_vy_s=NULL;
  d_dev_c=NULL; d_vx_c=NULL; d_vy_c=NULL;

  cudaFree(d_u_adhe);
  d_u_adhe=NULL;

  fout_vol.close();
  fout_nc.close();

  return 0;
}
