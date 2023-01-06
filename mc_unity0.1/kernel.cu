//#pragma comment( linker, "/subsystem:windows /entry:mainCRTStartup" )
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <tuple>
#include <random>
#include <iostream>
#include <fstream>
#include <cstdlib> 
#include <ctime>  

#define M 200//amount of a batch of photons
#define AM 1000
#define N int(100) // amount of vertex of model geometry

#define K int(1e5)
#define X 2 //kinds of participate medium
#define DELTA_T     1e-14  //1e-14
#define	PI          3.1415926
#define	LIGHTSPEED	2.997925E8 /* in vacuo speed of light [m/s] */
#define ALIVE       1   		/* if photon not yet terminated */
#define DEAD        0    		/* if photon is to be terminated */
#define THRESHOLD   1e-4		/* used in roulette */
#define CHANCE      0.1  		/* used in roulette */
#define ONE_MINUS_COSZERO 1.0E-12
/* If 1-cos(theta) <= ONE_MINUS_COSZERO, fabs(theta) <= 1e-6 rad. */
/* If 1+cos(theta) <= ONE_MINUS_COSZERO, fabs(PI-theta) <= 1e-6 rad. */
#define SIGN(x)           ((x)>=0 ? 1:-1)

struct Tissue {
    int    index;
    char   name[10];
    float miu_a;
    float miu_s;
    float g;
    float index_of_refraction;
};

__device__ thrust::tuple<float(*)[N][2]> bubbleSort(float(*ar)[N][2], int FLAG, int idx) {

    for (int i = 0; i < FLAG - 1; i++) {
        for (int j = 0; j < FLAG - 1 - i; j++) {
            if (ar[idx][j][0] > ar[idx][j + 1][0]) {
                float temp0 = ar[idx][j + 1][0];
                ar[idx][j + 1][0] = ar[idx][j][0];
                ar[idx][j][0] = temp0;
                float temp1 = ar[idx][j + 1][1];
                ar[idx][j + 1][1] = ar[idx][j][1];
                ar[idx][j][1] = temp1;
            }
        }
        //printf("this is %d:", i + 1);

    }

    return thrust::make_tuple(ar);
}

__device__ thrust::tuple<float, float, float> refra(float(*face1)[3], float(*face2)[3]
    , float(*face3)[3], float refra1, float refra2, float ux, float uy, float uz, float weight,
    float(*d1)[3], float(*d2)[3], float(*di)[3], float(*n1)[3], float(*n2)[3],
    float(*vrot)[3], float(*temp0)[3], float(*temp2)[3], int idx) {

    /*float* d1, * d2, * di, * n1, * n2, * vrot, * temp0, * temp2;
    cudaMalloc((void**)&d1, 3 * sizeof(float));
    cudaMalloc((void**)&d2, 3 * sizeof(float));
    cudaMalloc((void**)&di, 3 * sizeof(float));
    cudaMalloc((void**)&n1, 3 * sizeof(float));
    cudaMalloc((void**)&n2, 3 * sizeof(float));
    cudaMalloc((void**)&vrot, 3 * sizeof(float));
    cudaMalloc((void**)&temp0, 3 * sizeof(float));
    cudaMalloc((void**)&temp2, 3 * sizeof(float));*/
    int i;
    float tempn1, theta, theta1, theta2, temp1;

    for (i = 0;i < 3;i++) {
        d1[idx][i] = face1[idx][i] - face2[idx][i];
        d2[idx][i] = face2[idx][i] - face3[idx][i];
    }

    di[idx][0] = ux;/*light incident direction*/
    di[idx][1] = uy;
    di[idx][2] = uz;

    n1[idx][0] = d1[idx][1] * d2[idx][2] - d1[idx][2] * d2[idx][1];
    n1[idx][1] = d1[idx][2] * d2[idx][0] - d1[idx][0] * d2[idx][2];
    n1[idx][2] = d1[idx][0] * d2[idx][1] - d1[idx][1] * d2[idx][0];//normal line of the face : n1 = d1 × d2
    //----------------------------------------------------
    tempn1 = pow(n1[idx][0], 2) + pow(n1[idx][1], 2)
        + pow(n1[idx][2], 2);
    tempn1 = pow(tempn1, 0.5);
    n1[idx][0] = n1[idx][0] / tempn1;
    n1[idx][1] = n1[idx][1] / tempn1;
    n1[idx][2] = n1[idx][2] / tempn1;//make the norm of vector n1 to 1
    //-----------------------------------------------------
    if (n1[idx][0] * di[idx][0] + n1[idx][1] * di[idx][1] + n1[idx][2] * di[idx][2] < 0)
    {
        for (i = 0;i < 3;i++) n1[idx][i] = -n1[idx][i]; /*inverse normal to make sure
                                                        normal of the face is in the
                                                        same direction as light direction*/
    }

    n2[idx][0] = n1[idx][1] * di[idx][2] - n1[idx][2] * di[idx][1];
    n2[idx][1] = n1[idx][2] * di[idx][0] - n1[idx][0] * di[idx][2];
    n2[idx][2] = n1[idx][0] * di[idx][1] - n1[idx][1] * di[idx][0];/*normal line between the light direction and
                                                   the normal line of the face : n2 = n1 × light
                                                   direction*/
                                                   //----------------------------------------------------
    tempn1 = pow(n2[idx][0], 2) + pow(n2[idx][1], 2)
        + pow(n2[idx][2], 2);
    tempn1 = pow(tempn1, 0.5);
    n2[idx][0] = n2[idx][0] / tempn1;
    n2[idx][1] = n2[idx][1] / tempn1;
    n2[idx][2] = n2[idx][2] / tempn1;//make the norm of vector n1 to 1
    //-----------------------------------------------------
    theta1 = acos(n1[idx][0] * di[idx][0] + n1[idx][1] * di[idx][1] + n1[idx][2] * di[idx][2]);
    theta2 = asin(refra1 * sin(theta1) / refra2);

    if (theta2 - theta1 > 0) theta = theta2 - theta1;
    else theta = 2 * PI + theta2 - theta1;
    /* vector di rotate by angle theta about vector n2*/
    temp0[idx][0] = n2[idx][1] * di[idx][2] - n2[idx][2] * di[idx][1];
    temp0[idx][1] = n2[idx][2] * di[idx][0] - n2[idx][0] * di[idx][2];
    temp0[idx][2] = n2[idx][0] * di[idx][1] - n2[idx][1] * di[idx][0];/* temp = n2 × di*/

    temp1 = n2[idx][0] * di[idx][0] + n2[idx][1] * di[idx][1] + n2[idx][2] * di[idx][2];
    temp2[idx][0] = n2[idx][0] * temp1;
    temp2[idx][1] = n2[idx][1] * temp1;
    temp2[idx][2] = n2[idx][2] * temp1;

    for (i = 0;i < 3;i++) {
        vrot[idx][i] = di[idx][i] * cos(theta) + temp0[idx][i] * sin(theta)
            + temp2[idx][i] * (1 - cos(theta));
    }

    return thrust::make_tuple(vrot[idx][0], vrot[idx][1], vrot[idx][2]);
}

std::tuple<int, float, short int*, int, int(*)[3]> read_model(std::string directory) {
    int i, j, k;
    int flag, FLAG;
    int(*p)[3];
    static int p1[N][3];
    int Nbins;
    float length_voxel;
    short int* T;
    std::ifstream fin(directory, std::ios::binary);
    fin.read((char*)&Nbins, sizeof(int));
    fin.read((char*)&length_voxel, sizeof(float));
    T = (short int*)malloc(1 * pow(Nbins, 3) * sizeof(short int));
    for (i = 0; i < Nbins;i++) {
        for (j = 0;j < Nbins;j++) {
            for (k = 0;k < Nbins;k++) {

                fin.read((char*)&(*(T + i * Nbins * Nbins +
                    j * Nbins + k)), sizeof(short int));

            }
        }
    }
    fin.read((char*)&(FLAG), sizeof(int));
    for (flag = 0;flag < FLAG;flag++) {
        fin.read((char*)&(p1[flag][0]), sizeof(int));
        fin.read((char*)&(p1[flag][1]), sizeof(int));
        fin.read((char*)&(p1[flag][2]), sizeof(int));
    }
    fin.close();
    p = p1;
    return std::make_tuple(Nbins, length_voxel, T, FLAG, p);
    std::cout << "model written\n";
}


unsigned int Rand0(unsigned int randx)
{
    randx = randx * 1103515245 + 12345;
    return randx & 2147483647;
}

unsigned int Rand01(unsigned int randx)
{
    randx = randx * 1664525 + 1013904223;
    return randx & 4294967296;
}

__device__
unsigned int Rand1(unsigned int randx)
{
    randx = randx * 1103515245 + 12345;
    return randx & 2147483647;
}


__device__ thrust::tuple<float, bool, int, float, int, int, int, float(*)[K][5]>distance_to_interface(float x,
    float y, float z, float ux, float uy, float uz, float w,
    short int* T, float delta_t, int Nbins, float s,
    float length_voxel, Tissue a[X], int time, float(*co)[K][5], bool reset, int idx) {

    float xr = x, yr = y, zr = z;
    float s1 = 0;
    float weight = w;
    //static float co[K][5] = { 0 };
    //float(*p)[5];
    int flag = 1;
    int i = 0, j = 0;
    int if_refraction = 0;
    int if_scatter = 0;
    //static int time = 0;
    bool if_beyond;
    short int type0 = T[int(x / length_voxel) * Nbins * Nbins
        + int(y / length_voxel) * Nbins + int(z / length_voxel)];
    if_beyond = 0;
    float refraction = a[type0].index_of_refraction;
    float mua = a[type0].miu_a;
    float step = LIGHTSPEED * delta_t / refraction;
    float length_of_side = Nbins * length_voxel;

    if (reset) {
        time = 0;
        memset(co, 0, sizeof(co));
    }

    do {

        if (((xr / length_voxel) - 1) < 0 || ((yr / length_voxel) - 1) < 0
            || ((zr / length_voxel) - 1) < 0 ||
            ((xr / length_voxel) - 1) > Nbins - 1 ||
            ((yr / length_voxel) - 1) > Nbins - 1 ||
            ((zr / length_voxel) - 1) > Nbins - 1)
        {
            flag = 0;
            if_beyond = 1;
            //std::cout << "out of cube\n";
        }
        else
        {

            if (T[int((xr + step * ux) / length_voxel) * Nbins * Nbins +
                int((yr + step * uy) / length_voxel) * Nbins +
                int((zr + step * uz) / length_voxel)] == type0)
            {
                s1 += step;
                xr = xr + step * ux;
                yr = yr + step * uy;
                zr = zr + step * uz;
                weight = weight * (exp(-mua * step));
                if (i % 100 == 0) {
                    co[idx][j][0] = xr;
                    co[idx][j][1] = yr;
                    co[idx][j][2] = zr;
                    co[idx][j][3] = weight;
                    co[idx][j][4] = delta_t * time;
                    j++;
                }
                time++;
                i++;
            }
            else {
                //s1 += length_voxel;
                int la = 1;
                xr = xr + la * step * ux;
                yr = yr + la * step * uy;
                zr = zr + la * step * uz;
                if (i % 100 == 0) {
                    co[idx][j][0] = xr;
                    co[idx][j][1] = yr;
                    co[idx][j][2] = zr;
                    co[idx][j][3] = weight;
                    co[idx][j][4] = delta_t * time;
                    j++;
                }
                time++;
                i++;
                flag = 0;
                if_refraction = 1;
                //std::cout << "reach the interface\n";

            }
        }
        if (s1 >= s) {
            flag = 0;
            if_scatter = 1;
            //std::cout << "out of max length\n";
        }
        //std::cout << "flag = " << flag << "\n";
    } while (flag);
    //p = co;
    //std::cout << "i = " << i << "\n";
    return thrust::make_tuple(s1, if_beyond, j,
        weight, if_refraction, if_scatter, time, co);
}

__device__ thrust::tuple<float(*)[K][5], int> record_path(float(*c0)[K][5],
    int k, float(*path)[K][5], int k0, int reset, int idx) {
    //auto static  path = new float[K][5];

    //static int k0 = 0;
    int i, j;
    if (reset == 0) {

        for (i = 0;i < 5;i++) {
            //path[i].resize((k0 + k) * sizeof(float));
            //if(k0 ==0) path[0][i] = k;

            for (j = 0;j < k;j++) {
                path[idx][k0 + j + 1][i] = c0[idx][j][i];
            }
            //c00[i] = c0[j][i];
        }

    }
    else if (reset == 1) {

        memset(path, 0, sizeof(path));
        k0 = 0;

    }
    k0 = k0 + k;
    for (i = 0;i < 5;i++) path[idx][0][i] = k0;
    return thrust::make_tuple(path, k0);

    //k0 = k0 + k;

}



__device__ thrust::tuple< float(*)[K][5] >  montecarlo
(short int* T, int Nbins, float length_voxel, Tissue a[X], int FLAG,
    int(*vertex)[3], float(*path2_gpu)[K][5], float(*c0)[K][5], int rndseed, int idx
    , float(*diss)[N][2], float(*face1)[3], float(*face2)[3],
    float(*face3)[3], float(*d1)[3], float(*d2)[3], float(*di)[3], float(*n1)[3], float(*n2)[3],
    float(*vrot)[3], float(*temp0)[3], float(*temp2)[3]) {

    /* propagation parameters */
    float	x, y, z;        /* photon position */
    float	ux, uy, uz;     /* photon trajectory as cosines */
    double  uxx, uyy, uzz;	/* temporary values used during spin */
    double	s;              /* step sizes. s = -log(rnd)/mus [cm] */
    double  sleft = 0;          /* dimensionless */
    double	costheta;       /* cos(theta) */
    double  sintheta;       /* sin(theta) */
    double	cospsi;         /* cos(psi) */
    double  sinpsi;         /* sin(psi) */
    double	psi;            /* azimuthal angle */
    long	i_photon;       /* current photon */
    double	w;              /* photon weight */
    double	absorb;         /* weighted deposited in a step due to absorption */
    short   photon_status;  /* flag = alive=1 or dead=0 */

    /* other variables */
    double	mua;            /* absorption coefficient [cm^-1] */
    double	mus;            /* scattering coefficient [cm^-1] */
    double	g;              /* anisotropy [-] */
    double  refraction;     /* index of refraction*/
    double	nphotons;       /* number of photons in simulation */

    /* dummy variables */
    float  rnd;            /* assigned random value 0-1 */

    int i_voxel;
    bool flag1 = 0;
    double 	temp;           /* dummy variable */

    /* mcxyz bin variables */
    //float   length_voxel = 0.002;/*bins size, unit : [m] */
    //float   step = 0.5;
    //float   stepsize = length_voxel * step;
    double   delta_t = DELTA_T;

    float s1 = 0;
    float s3; //distance from initial point to the current position
    //int s_block;
    int if_refraction = 0;
    int if_scatter = 0;

    //float c0[K][5];
    //auto c0 = new float[K][5];
    //cudaMemset(c0, 0, K * 5 * sizeof(float));
    //extern __shared__ float c0[K][5];


    int k0 = 0;

    int if_beyond, k;
    int i, j;
    int rndflag = 0;
    //auto rand = new float[COUNT];
    int n_rand = rndseed;

    //float (*dis)[2];
    //cudaMalloc((void**) & dis, N * 2 * sizeof(float));
    //float(*dis2)[2]; // all surface point before and after bubble sort
    ////float face1[3], face2[3], face3[3]; // coordinates of the 3 nearest point
    //float* face1;
    //float* face2;
    //float* face3;
    //cudaMalloc((void**)&face1, 3 * sizeof(float));
    //cudaMalloc((void**)&face2, 3 * sizeof(float));
    //cudaMalloc((void**)&face3, 3 * sizeof(float));
    int pos; //position of the minimum 3 point near the light penetration point
    float refra0, refra1; //refraction index before and after refraction

    //thrust::tie(n_rand,rnd,rand) = randnum(n_rand,rand,0)

    nphotons = M - 1; // will be updated to achieve desired run time, time_min.
    i_photon = -1;


    i_photon += 1;				/* increment photon count */
    w = 1.0;                    /* set photon weight to one */
    photon_status = ALIVE;      /* launch an alive photon */

    x = 0.01 * Nbins * length_voxel;
    y = 0.5 * Nbins * length_voxel;
    z = 0.5 * Nbins * length_voxel;
    ux = 1;
    uy = 0;
    uz = 0;
    sleft = 0;
    //std::cout << "\n\n\nphoton : " << i_photon << " launched\n";
    int loop = 0;
    int time = 0;

    /*distance_to_interface(x, y, z, ux, uy, uz, w, T,
        delta_t, Nbins, 0, length_voxel, a,time,c0, 1);*/


    do {
        loop++;


        if (sleft == 0 || flag1 == 1) {

            //thrust::tie(n_rand, rnd, rand) = randnum(n_rand, rand, 0);
            n_rand = Rand1(n_rand);
            rnd = float(n_rand) / 2147483647;

            sleft = -log(rnd);

            flag1 = 0;
        }
        else {
            i_voxel = int(x / length_voxel) * Nbins * Nbins
                + int(y / length_voxel) * Nbins + int(z / length_voxel);
            mua = a[T[i_voxel]].miu_a;
            mus = a[T[i_voxel]].miu_s;
            g = a[T[i_voxel]].g;
            refraction = a[T[i_voxel]].index_of_refraction;

            s = sleft / (mus * (1 - g) + mua);  //[m]
            //s = sleft / (mus );  //[m]

            thrust::tie(s1, if_beyond, k, w,
                if_refraction, if_scatter, time, c0) =
                distance_to_interface(x, y, z, ux, uy, uz, w, T,
                    delta_t, Nbins, s, length_voxel, a, time, c0, 0, idx);
            //std::cout << "\n\nw = " << w << "\n\n";

            /*i_voxel = int(round((x / length_voxel) * Nbins * Nbins +
                (y / length_voxel) * Nbins + (z / length_voxel)));*/
            refra0 = a[T[i_voxel]].index_of_refraction;
            thrust::tie(path2_gpu, k0) = record_path(c0, k, path2_gpu, k0, 0, idx);

            /*std::cout << "c0.time : \n" ;
            for (i = 0;i < 10;i++) {
                std::cout << c0[i][4]<<"\n";
            }
            std::cout << "\n\n";*/
            //path1[i_photon] = path0;


            if (if_beyond == 1) { photon_status = DEAD; }
            if (s1 >= s) { s3 = s; flag1 = 1; }
            else s3 = s1;

            //std::cout << "s3 = " << s3 << " " << "s1 = " << s1 << "\n";

            k--;
            //if (k > 0) k--;
            /*int la = 1;
            x = c0[k][0] + la * length_voxel * ux;
            y = c0[k][1] + la * length_voxel * uy;
            z = c0[k][2] + la * length_voxel * uz;*/
            x = c0[idx][k][0];
            y = c0[idx][k][1];
            z = c0[idx][k][2];

            /*i_voxel = int(floor(((x / length_voxel)-1) * Nbins * Nbins +
                ((y / length_voxel)-1) * Nbins + ((z / length_voxel)-1)));*/
            i_voxel = int(x / length_voxel) * Nbins * Nbins
                + int(y / length_voxel) * Nbins + int(z / length_voxel);

            //std::cout << "xyz=" << x << y << z << "\n";
            if (i_voxel > pow(Nbins, 3) - 1) photon_status = DEAD;
            if (x <= 0 || y <= 0 || z <= 0 || x >=
                Nbins * length_voxel || y >= Nbins * length_voxel ||
                z >= Nbins * length_voxel) {
                photon_status = DEAD;
            }
            if (i_voxel > sizeof(T) / sizeof(T[0])) i_voxel = sizeof(T) / sizeof(T[0]);
            refra1 = a[T[i_voxel]].index_of_refraction;

            //sleft = sleft - s3 * mus;
            sleft = sleft - s3 * (mus * (1 - g) + mua);

        }

        if (if_refraction == 1) {
            for (i = 0;i < FLAG;i++) {
                diss[idx][i][0] = pow((pow(x - length_voxel * vertex[i][0], 2)
                    + pow(y - length_voxel * vertex[i][1], 2)
                    + pow(z - length_voxel * vertex[i][2], 2)), 0.5);
                diss[idx][i][1] = i;
            }

            thrust::tie(diss) = bubbleSort(diss, FLAG, idx);
            pos = diss[idx][N - FLAG][1];
            //std::cout << "pos = " << pos << std::endl;
            for (i = 0;i < 3;i++) {
                face1[idx][i] = length_voxel * vertex[pos][i];
            }
            pos = diss[idx][N - FLAG + 1][1];
            for (i = 0;i < 3;i++) {
                face2[idx][i] = length_voxel * vertex[pos][i];
            }
            pos = diss[idx][N - FLAG + 2][1];
            for (i = 0;i < 3;i++) {
                face3[idx][i] = length_voxel * vertex[pos][i];
            }
            thrust::make_tuple(ux, uy, uz) =
                refra(face1, face2, face3, refra0, refra1, ux, uy, uz, w,
                    d1, d2, di, n1, n2, vrot, temp0, temp2, idx);

        }

        if (if_scatter == 1) {
            /* sample for costheta */
            n_rand = Rand1(n_rand);
            rnd = float(n_rand) / 2147483647;
            if (g == 0.0)
                costheta = 2.0 * rnd - 1.0;
            else {
                temp = (1.0 - g * g) / (1.0 - g + 2 * g * rnd);
                costheta = (1.0 + g * g - temp * temp) / (2.0 * g);
            }
            sintheta = sqrt(1.0 - costheta * costheta); /* sqrt() is faster than sin(). */

            /* sample psi. */
            n_rand = Rand1(n_rand);
            rnd = float(n_rand) / 2147483647;
            psi = 2.0 * PI * rnd;
            //psi = 2.0 * PI * RandomNum;
            cospsi = cos(psi);
            if (psi < PI)
                sinpsi = sqrt(1.0 - cospsi * cospsi);     /* sqrt() is faster than sin(). */
            else
                sinpsi = -sqrt(1.0 - cospsi * cospsi);

            /* new trajectory. */
            if (1 - fabs(uz) <= ONE_MINUS_COSZERO) {      /* close to perpendicular. */
                uxx = sintheta * cospsi;
                uyy = sintheta * sinpsi;
                uzz = costheta * SIGN(uz);   /* sign() is faster than division. */
            }
            else {					/* usually use this option */
                temp = sqrt(1.0 - uz * uz);
                uxx = sintheta * (ux * uz * cospsi - uy * sinpsi) / temp + ux * costheta;
                uyy = sintheta * (uy * uz * cospsi + ux * sinpsi) / temp + uy * costheta;
                uzz = -sintheta * cospsi * temp + uz * costheta;
            }

            /* update trajectory */
            ux = uxx;
            uy = uyy;
            uz = uzz;

            //std::cout << "new direction = " << ux << "," << uy << "," << uz << "\n";

        }
        /**** check roulette
             if photon weight below threshold, then terminate photon using roulette technique.
             photon has chance probability of having its weight increased by factor of 1/chance,
             and 1-chance probability of terminating.
             *****/
        if (w < THRESHOLD) {
            n_rand = Rand1(n_rand);
            rnd = float(n_rand) / 2147483647;
            if (rnd <= CHANCE)
                w /= CHANCE;
            else
            {
                photon_status = DEAD;
                //std::cout << "i_photon = " << i_photon << "\n";
                //std::cout << "coordinate = " << x << "," << y << "," << z << "\n";

            }
        }






    } while (photon_status == ALIVE);  /* end step_check_hop_spin */
    /* if alive, continue propagating */
    /* if photon dead, then launch new photon. */

    /*std::cout << "path0.time : \n" ;
            for (i = 0;i < 100;i++) {
                std::cout << path0[i][4]<<"\n";
            }*/
            /*for (i = 0;i < K;i++) {
                for (j = 0;j < 5;j++) {
                    path1[i_photon][i][j] = path0[i][j];
                }
            }*/

            //record_path(c0, k, path0,k0,1);




    return thrust::make_tuple(path2_gpu);

}

__global__ void simulate(short int* T, int Nbins, float length_voxel, Tissue a[X], int FLAG,
    int(*vertex)[3], float(*path2_gpu)[K][5], float(*c0)[K][5],
    int* rndseed_gpu, float(*diss)[N][2], float(*face1)[3], float(*face2)[3],
    float(*face3)[3], float(*d1)[3], float(*d2)[3], float(*di)[3], float(*n1)[3], float(*n2)[3],
    float(*vrot)[3], float(*temp0)[3], float(*temp2)[3]) {

    int i = threadIdx.x;
    //  int i = 0;
    int j, k;
    if (i < M) {

        thrust::tie(path2_gpu) = montecarlo(T, Nbins, length_voxel, a, FLAG,
            vertex, path2_gpu, c0, rndseed_gpu[i], i, diss, face1, face2, face3,
            d1, d2, di, n1, n2, vrot, temp0, temp2);




    }
    //return thrust::make_tuple(path2_gpu);
}

void write_path(std::string directory) {
    std::ofstream fout(directory, std::ios::binary);
    int i, j, k;
    int I, J, L;
    I = AM;
    //I = 2;
    J = 5;
    L = 50;
    fout.write((char*)&(I), sizeof(int));
    fout.write((char*)&(J), sizeof(int));
    fout.write((char*)&(L), sizeof(int));
    //for (i = 0;i < M;i++) {
    //    for (j = 0;j < J;j++) {
    //        for (k = 0;k < path[i][j].size();k++) {
    //            fout.write((char*)&path[i][j][k], sizeof(float));
    //            //std::cout << path[i][j][k] << "\n";
    //        }
    //        fout.write("end", 4);
    //    }
    //}

    fout.close();
}

void write_path3(std::string directory) {
    std::ofstream fout(directory, std::ios_base::binary |
        std::ios_base::app | std::ios_base::out);

    int I;
    I = 800;
    fout.seekp(0, SEEK_SET);
    fout.write((char*)&(I), sizeof(int));

    fout.close();
}

void write_path2(std::vector< std::vector<float> >(*path), std::string directory) {
    using namespace std;
    //std::fstream fout(directory, std::ios::binary| ios::out | ios::in);
    //ofstream fout(directory, ios::out | ios::binary);
    //ofstream fout(directory, ios_base::app);
    std::ofstream fout(directory, std::ios_base::binary | ios_base::app | ios_base::out);
    //std::ofstream fout(directory, ios_base::app | ios_base::out);

    int i, j, k;
    int I, J, L;
    I = M;
    J = 5;
    L = 50;
    //std::cout << "size = " << path[0][0][0];
    for (i = 0;i < M;i++) {
        for (j = 0;j < J;j++) {
            for (k = 0;k < path[i][j].size();k++) {
                fout.write((char*)&path[i][j][k], sizeof(float));
                //std::cout << path[i][j][k] << "\n";
            }
            fout.write("end", 4);
        }
    }

    fout.close();

}

std::vector< std::vector<float> >* array2vector(float(*path1)[K][5],
    bool reset) {
    int i, j, l;
    int k;
    static std::vector< std::vector<float> > path[M];
    if (reset == 0) {
        for (i = 0;i < M;i++) {
            for (j = 0;j < 5;j++) {
                k = path1[i][0][j];
                path[i].resize(5 * sizeof(float));
                for (l = 1;l < k;l++) {
                    path[i][j].push_back(path1[i][l][j]);
                }
            }

        }
    }
    else {
        for (i = 0;i < M;i++) path[i].clear();
    }

    //std::cout << "size = " << path[0][4].size();
    //std::cout << "010101\n";
    return path;
}

std::tuple<float,float,float> read_para(std::string dic) {
    float a[4];
    std::ifstream fin(dic, std::ios::binary);

    fin.read((char*)&a[0], sizeof(float));
    fin.read((char*)&a[1], sizeof(float));
    fin.read((char*)&a[2], sizeof(float));
    fin.read((char*)&a[3], sizeof(float));
    return std::make_tuple(a[1],a[2],a[3]);
}

int main()
{
    clock_t tStart = clock();
    Tissue a[X];
    //a[0] = { 0,"air",0.0001,1,1.0 ,1};
    float parax[3];
    char directory2[] = "C:\\Users\\Administrator\\source\\data\\para.bin";
    std::tie(parax[0], parax[1], parax[2]) = read_para(directory2);
    a[0] = { 0,"air",     0.01,  0.01,   0 ,   1 };
    a[1] = { 1,"midium1", parax[0],  parax[1], parax[2] ,1.333};
    int* seedofseed = (int*)malloc(sizeof(int) * int(AM / M));

    int Nbins;
    float length_voxel;
    int(*vertex)[3];
    int FLAG;
    int i, j;
    short int* T1;


    int threadsPerBlock = M;
    int numBlocks = 1;


    cudaError_t c;

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::string directory0 = "C:\\Users\\Administrator\\source\\data\\model_slab.dat";
    std::string directory1 = "C:\\Users\\Administrator\\source\\data\\path_slab.dat";
    std::vector< std::vector<float> >(*path);
    //std::vector< std::vector<float> > path_AM[AM];

    std::tie(Nbins, length_voxel, T1,
        FLAG, vertex) = read_model(directory0);


    write_path(directory1);

    /*seedofseed[0] = 1145141919;
    for (i = 0;i < int(AM/M)-1;i++) {
        seedofseed[i + 1] = Rand01(seedofseed[i]);
        std::cout << "dis = " << seedofseed[i+1] << "\n";
    }*/
    /*for (i = 0;i < int(AM / M);i++) {
        seedofseed[i] = dis(gen);
        std::cout << "dis = " << seedofseed[i] << "\n";
    }*/

    for (j = 0;j < (AM / M);j++) {

        auto path2 = new float[M][K][5];
        int* rndseed = (int*)malloc(sizeof(int) * M);

        short int* T1_gpu;
        Tissue* a_gpu;
        int(*vertex_gpu)[3];
        float(*path2_gpu)[K][5];
        float(*c0)[K][5];
        int* rndseed_gpu;
        float(*diss)[N][2];
        float(*face1)[3];
        float(*face2)[3];
        float(*face3)[3];
        float(*d1)[3], (*d2)[3], (*di)[3], (*n1)[3], (*n2)[3], (*vrot)[3], (*temp0)[3], (*temp2)[3];

        cudaMalloc((void**)&diss, M * N * 2 * sizeof(float));
        cudaMalloc((void**)&face1, M * 3 * sizeof(float));
        cudaMalloc((void**)&face2, M * 3 * sizeof(float));
        cudaMalloc((void**)&face3, M * 3 * sizeof(float));
        cudaMalloc((void**)&d1, M * 3 * sizeof(float));
        cudaMalloc((void**)&d2, M * 3 * sizeof(float));
        cudaMalloc((void**)&di, M * 3 * sizeof(float));
        cudaMalloc((void**)&n1, M * 3 * sizeof(float));
        cudaMalloc((void**)&n2, M * 3 * sizeof(float));
        cudaMalloc((void**)&vrot, M * 3 * sizeof(float));
        cudaMalloc((void**)&temp0, M * 3 * sizeof(float));
        cudaMalloc((void**)&temp2, M * 3 * sizeof(float));

        cudaMalloc(&T1_gpu, pow(Nbins, 3) * sizeof(short int));
        cudaMalloc(&a_gpu, sizeof(a));
        cudaMalloc(&vertex_gpu, N * 3 * sizeof(int));
        cudaMalloc(&path2_gpu, M * K * 5 * sizeof(float));
        cudaMalloc(&c0, M * K * 5 * sizeof(float));
        cudaMalloc(&rndseed_gpu, M * sizeof(int));

        /*rndseed[0] = dis(gen)* 2147483647;
        for (i = 0;i < M-1;i++) {
            rndseed[i + 1] = Rand0(rndseed[i]);

        }*/
        for (i = 0;i < M;i++) {
            rndseed[i] = dis(gen) * 2147483647;

        }

        cudaMemcpy(T1_gpu, T1, pow(Nbins, 3) * sizeof(short int), cudaMemcpyHostToDevice);
        cudaMemcpy(a_gpu, a, sizeof(a), cudaMemcpyHostToDevice);
        cudaMemcpy(vertex_gpu, vertex, N * 3 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(rndseed_gpu, rndseed, M * sizeof(int), cudaMemcpyHostToDevice);

        simulate << <1, M >> > (T1_gpu, Nbins, length_voxel,
            a_gpu, FLAG, vertex_gpu, path2_gpu, c0, rndseed_gpu, diss, face1, face2, face3,
            d1, d2, di, n1, n2, vrot, temp0, temp2);
        //thrust::tie(path2) = simulate(T1, Nbins, length_voxel, a, FLAG, vertex, path0,path2);
        c = cudaDeviceSynchronize();
        std::cout << "c : " << c << "\n";

        cudaMemcpy(path2, path2_gpu, M * K * 5 * sizeof(float), cudaMemcpyDeviceToHost);
        path = array2vector(path2, 0);
        //memcpy(&*(path_AM + j * M), path,sizeof(path)*M);

        write_path2(path, directory1);

        array2vector(path2, 1);

        cudaFree(T1_gpu);
        cudaFree(a_gpu);
        cudaFree(vertex_gpu);
        cudaFree(path2_gpu);
        cudaFree(c0);
        cudaFree(rndseed_gpu);
        cudaFree(diss);
        cudaFree(face1);cudaFree(face2);cudaFree(face3);
        cudaFree(d1);cudaFree(d2);cudaFree(di);
        cudaFree(n1);cudaFree(n2);cudaFree(vrot);cudaFree(temp0);cudaFree(temp2);
        free(path2);
        free(rndseed);
    }
    free(seedofseed);
    
    

    printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
    return 0;
}
