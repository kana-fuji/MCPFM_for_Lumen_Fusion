#ifndef CONFIG_CUH
#define CONFIG_CUH

namespace PARAM
{

  typedef struct {
    int imax, jmax, kmax;
    float dx, dy, dz;

    int cimax, cjmax, ckmax;
    
    float dt;
    int nmax;

    float dts;
    
  }psys;

  typedef struct {
    float D_u;
    float tau_u;
    float alpha;
    float beta;
    float gamma;
    float eta;
    float V;
    float vd;
    float v;
    float gamma_curv;
    float gamma_curv1;
    float gamma_curv2;

    float tau_V;
    float alpha_V;

    float D_s;
    float tau_s;
    float xi;
    float beta_s;
    float gamma_s;
    float eta_s;
    float alpha_s;
    float p_st,v_t;
    float alpha_sg;
    float targetVl;

    float D_c;
    float tau_c;
    float beta_cu;
    float beta_cs;
    float gamma_c;
    float eta_cu;
    float eta_cs;
    float xi_c;
    float alpha_c;  // 20220520 Bulk modulus of medium

    float ep_d;
    float rho0;
    float rhoe;
    float mu;
    float sigma;
    float ls;

    float D_p;
    float tau_p;
    float Vp;
    float alpha_p;
    float eta_ps;
    float gamma_p;
    float C_p;
    float w_p;
    float p_r;
    float k_p;
    float p_th;
    float l_anti;

    float D_pol;
    float tau_pol;
    float gamma_pol;
  }param;

  typedef struct {
    float v;
    float targetv;
    float vp;

    int cimin, cjmin, ckmin;
    float Gx, Gy, Gz;

    float tauV;
  }cells;

  typedef struct {
    double dtargetv;
  }dtv;

  typedef struct {
    int   cnum;
  }num;

  typedef struct {
    int di, dj, dk;
  }com;

  // typedef struct {
  //   int imin, jmin, kmin;
  //   float v;
  // }vlumen;

}

#endif
