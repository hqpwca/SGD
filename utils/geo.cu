#include "geo.hh"

void multm(double *a, double *b, int x, int y, int z)
{
    double *c;
    int i,j,k,m;

    c = (double *) malloc(x*z*sizeof(double));

    // multiply
    for (i=0;i<x; i++) {
        for (j=0; j<z; j++) {
            m = i*z + j;
            c[m] = 0.0;
            for (k=0; k<y; k++) {
                c[m] += a[i*y + k]*b[k*z + j];
            }
        }
    }

    // copy
    for (i=0;i<x;i++) {
        for (j=0; j<z; j++) {
            m = i*z + j;
            b[m] = c[m];
        }
    }
}

void invert3(double *m)
{
    int i;
    double n[] = {0,0,0,0,0,0,0,0,0};

    // invert and transpose
    double det = m[0]*((m[4]*m[8]) - (m[5]*m[7])) - m[1]*((m[3]*m[8]) - (m[5]*m[6])) + m[2]*((m[3]*m[7]) - (m[4]*m[6]));

    // prevent this from blowing up at multiples of 45 degrees
    if (fabs(det) < 1e-10) {
        n[0] = 1, n[4] = 1, n[8] = 1;
    }
    else {
        n[0] =    ((m[4]*m[8]) - (m[5]*m[7]))/det;
        n[1] = -1*((m[1]*m[8]) - (m[2]*m[7]))/det;
        n[2] =    ((m[1]*m[5]) - (m[2]*m[4]))/det;
        n[3] = -1*((m[3]*m[8]) - (m[5]*m[6]))/det;
        n[4] =    ((m[0]*m[8]) - (m[2]*m[6]))/det;
        n[5] = -1*((m[0]*m[5]) - (m[2]*m[3]))/det;
        n[6] =    ((m[3]*m[7]) - (m[4]*m[6]))/det;
        n[7] = -1*((m[0]*m[7]) - (m[1]*m[6]))/det;
        n[8] =    ((m[0]*m[4]) - (m[1]*m[3]))/det;
    }

    // copy
    for (i=0;i<9;i++) m[i] = n[i];
}

__host__ void generate_projection_matrix(double *pm, float *pmi, const double *src, const double *dtv, const double *puv, const double *pvv, const double uc, const double vc, const int3 n3xyz, const double3 d3xyz) {
    double vec[3];
    double k;
    double norm[3];
    int i,j,l;

    norm[0] = puv[1]*pvv[2] - puv[2]*pvv[1];
    norm[1] = puv[2]*pvv[0] - puv[0]*pvv[2];
    norm[2] = puv[0]*pvv[1] - puv[1]*pvv[0];

    // std::cout << norm[0] << ' ' << norm[1] << ' ' << norm[2] << std::endl << std::endl;

    k = 0.0f;
    for (i=0;i<3;i++) k += norm[i]*(dtv[i]-src[i]);

    for (i=0;i<3;i++) vec[i] = src[i] - dtv[i] + (uc*puv[i]) + (vc*pvv[i]) + 1;

    // start with identity
    double pm4[] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1};

    // translation 
    double trans[] = {
        1,0,0,-src[0],
        0,1,0,-src[1],
        0,0,1,-src[2],
        0,0,0,1};

    multm(trans,pm4,4,4,4); // vector from source to object

    // divide by dot product
    double divdot[] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        norm[0],norm[1],norm[2],0};

    multm(divdot,pm4,4,4,4); //get ratio of S->O and S->P by division of 2 dot product with vector perpendicular to detector plane


    // scale
    double scale[] = {
        k,0,0,0,
        0,k,0,0,
        0,0,k,0,
        0,0,0,1};

    multm(scale,pm4,4,4,4); // vector from source to projection point on detector plane

    // translation
    double trans2[] = {
        1,0,0,vec[0],
        0,1,0,vec[1],
        0,0,1,vec[2]};

    multm(trans2,pm4,4,4,4); // vector from detection top left to projection point

    // detector misalignment
    double align[] = {
        puv[0],pvv[0],1,
        puv[1],pvv[1],1,
        puv[2],pvv[2],1};
    
    // printf("align   :(%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf)\n", align[0], align[1], align[2], align[3], align[4], align[5], align[6], align[7], align[8]);

    invert3(align);

    // printf("align^-1:(%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf)\n", align[0], align[1], align[2], align[3], align[4], align[5], align[6], align[7], align[8]);

    multm(align,pm4,3,3,4); // (x, y, z) vector to (u, v) pixel

    for (i=0;i<3;i++) {
        for (j=0;j<4;j++) {
            l = i*4 + j;
            pm[l] = pm4[l]/pm4[11];
        }
    }

    // copy temporary matrix to output, scale by value in lower right corner

    int nx,ny,nz;
    float dx,dy,dz,ox,oy,oz;

    dx = d3xyz.x;
    nx = n3xyz.x;
    dy = d3xyz.y;
    ny = n3xyz.y;
    dz = d3xyz.z;
    nz = n3xyz.z;

    ox = (nx-1)*dx/2.0f; 
    oy = (ny-1)*dy/2.0f;
    oz = (nz-1)*dz/2.0f;

    // printf("NXY: %d, %d, %d, DXY: %f, %f, %f, OXY: %f, %f, %f\n", nx,ny,nz, dx, dy, dz, ox, oy, oz);

    // scale for multiplication by int indices
    pmi[0] = dx*pm[0]; pmi[1] = dy*pm[1]; pmi[2] = dz*pm[2]; pmi[3] = pm[3] - ox*pm[0] - oy*pm[1] - oz*pm[2];
    pmi[4] = dx*pm[4]; pmi[5] = dy*pm[5]; pmi[6] = dz*pm[6]; pmi[7] = pm[7] - ox*pm[4] - oy*pm[5] - oz*pm[6];
    pmi[8] = dx*pm[8]; pmi[9] = dy*pm[9]; pmi[10] = dz*pm[10]; pmi[11] = pm[11] - ox*pm[8] - oy*pm[9] - oz*pm[10];
}

GeoData::GeoData(int nx, int ny, int nz, int nu, int nv, int np, double dx, double dy, double dz, double du, double dv) :
    srcs(np, 3), puvs(np, 3), pvvs(np, 3), dtvs(np, 3), ucs(np, 1), vcs(np, 1), pms(np, 12), pmis(np, 12),
    lsds(np, 1), lsos(np, 1) {
    nxyz = make_int3(nx, ny, nz);
    dxyz = make_double3(dx, dy, dz);
    this->np = np;
    nuv = make_int2(nu, nv);
    duv = make_double2(du, dv);

    srcs.allocateHostMemory();
    puvs.allocateHostMemory();
    pvvs.allocateHostMemory();
    dtvs.allocateHostMemory();
    ucs.allocateHostMemory();
    vcs.allocateHostMemory();
    pms.allocateHostMemory();
    pmis.allocateHostMemory();
    lsds.allocateHostMemory();
    lsos.allocateHostMemory();
}

GeoData::~GeoData() {

}

void GeoData::initialize_projection_matrix() {
    for (int p=0; p<np; p++) {
        int i = p * 3;
        int j = p * 12;

        //std::cout << p << std::endl;
        generate_projection_matrix(pms[j], pmis[j], srcs[i], dtvs[i], puvs[i], pvvs[i], *ucs[p], *vcs[p], nxyz, dxyz);
    }

    pms.allocateCudaMemory();
    pms.copyHostToDevice();
    pmis.allocateCudaMemory();
    pmis.copyHostToDevice();
}

void rotate(double *vec, double *res, double angle) {
    res[0] = vec[0] * cos(angle) - vec[1] * sin(angle);
    res[1] = vec[0] * sin(angle) + vec[1] * cos(angle);
    res[2] = vec[2];
}

void GeoData::geo_init_example(double lsd, double lso,  double start_angle, double end_angle) {
    double dangle = (end_angle - start_angle) / (np - 1);

    double src[3] = {-lso, 0.0f, 0.0f};
    double dtv[3] = {lsd-lso, 0.0f, 0.0f};
    double puv[3] = {0.0f, duv.x, 0.0f};
    double pvv[3] = {0.0f, 0.0f, duv.y};

    for(int i = 0; i < np; ++i) {
        double beta = start_angle + i * dangle;

        rotate(src, srcs[i*3], beta);
        rotate(dtv, dtvs[i*3], beta);
        rotate(puv, puvs[i*3], beta);
        rotate(pvv, pvvs[i*3], beta);

        *lsds[i] = lsd;
        *lsos[i] = lso;
        *ucs[i] = double(nuv.x) / 2.0;
        *vcs[i] = double(nuv.y) / 2.0;
    }
}

void GeoData::geo_init_angles(double lsd, double lso, double *angles, double *dz, double *drho) {

    double src[3] = {-lso, 0.0f, 0.0f};
    double dtv[3] = {lsd-lso, 0.0f, 0.0f};
    double puv[3] = {0.0f, duv.x, 0.0f};
    double pvv[3] = {0.0f, 0.0f, duv.y};
    double _src[3] = {-lso, 0.0f, 0.0f};

    for(int i = 0; i < np; ++i) {
        long double beta = angles[i];
        
        if(dz != nullptr)
            src[2] = dz[i];
        else
            src[2] = 0.0;

        if(drho != nullptr)
            src[0] = _src[0] + drho[i];
        else
            src[0] = _src[0];


        if(abs(sin(beta) - cos(beta)) < 1e-7)
            beta = (sin(beta)>0)?PI/4:-PI*3/4;
        else if (abs(sin(beta) + cos(beta)) < 1e-7)
            beta = (sin(beta)>0)?PI*3/4:-PI/4;

        rotate(src, srcs[i*3], beta);
        rotate(dtv, dtvs[i*3], beta);
        rotate(puv, puvs[i*3], beta);
        rotate(pvv, pvvs[i*3], beta);

        *lsds[i] = lsd;
        *lsos[i] = lso;
        *ucs[i] = double(nuv.x) / 2.0;
        *vcs[i] = double(nuv.y) / 2.0;
    }
}

//Helical cylindrical projection, add one projection angle with DICOM-CT-PD geometry
// void GeoData::geo_add_projection_hc(double *src, double *)