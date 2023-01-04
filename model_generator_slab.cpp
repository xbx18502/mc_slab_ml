// model_generator_cube.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <matplot/matplot.h>
#include <tuple>
#include <random>
#include <iostream>
#include <cstdlib> 
#include <ctime> 

#define Nb 300;  /*Nbins*/
#define N int(4096) //nums of vertex

struct Tissue {
    int    index;
    char   name[10];
    float miu_a;
    float miu_s;
    float g;
    float index_of_refraction;
};

short int* create3Dmodel(int Nbins, float binsize) {

    
    //double binsize = 0.1e-3; 	//% size of each bin, eg. [cm] or [mm]
//    double T[Nbins][Nbins][Nbins][3];
    short int* T;
    int i, j, k, f = 0;
    int A = 0, B = 0, C = 0;
    /*user defines*/
    /*-------------------------------------*/
    float d;
    d = 20e-3;
    /*------------------------------------*/
    float x0, x1;
    x0 = Nbins * 0.75;
    x1 = Nbins * 0.75 - d/binsize;
    

    T = (short int*)malloc(1 * pow(Nbins, 3) * sizeof(short int));

    memset(T, 0, sizeof(short) * pow(Nbins, 3));

    printf("success00\n");
    for (i = 0; i < Nbins;i++) {
        for (j = 0;j < Nbins;j++) {
            for (k = 0;k < Nbins;k++) {


                if (x1<i && i<=x0) {
                    *(T + i * Nbins * Nbins + j * Nbins + k) = 1;

                    A++;

                }
                else { ; }
                    
                //



            }
            //            printf("success02\n");
        }
        //printf("success03\n");
    }

    //    printf("success04\n");
    //    i = 0; j = Nbins*0.5; k = Nbins*0.5; l = Nbins*0.5;
    printf("%d,%d,%d\n", A, B, C);
    //    printf("%f,%f,%f\n", *(T + i * Nbins * Nbins * Nbins + j * Nbins * Nbins + k * Nbins + l-1));

    return (T);
}

short int* create3Dmodel_lr(int Nbins, int mag, float binsize) { //lr: low resolution

    //double binsize = 0.1e-3*mag; 	//% size of each bin, eg. [cm] or [mm]
//    double T[Nbins][Nbins][Nbins][3];
    short int* T;
    int i, j, k, f = 0;
    int A = 0, B = 0, C = 0;
    /*user defines*/
    /*-------------------------------------*/
    float d;
    d = 20e-3;
    /*------------------------------------*/
    float x0, x1;
    x0 = Nbins * 0.75;
    x1 = Nbins * 0.75 - d / binsize;


    T = (short int*)malloc(1 * pow(Nbins, 3) * sizeof(short int));

    memset(T, 0, sizeof(short) * pow(Nbins, 3));

    printf("success00\n");
    for (i = 0; i < Nbins;i++) {
        for (j = 0;j < Nbins;j++) {
            for (k = 0;k < Nbins;k++) {


                if (x1/mag < i && i <= x0/mag) {
                    *(T + i * Nbins * Nbins + j * Nbins + k) = 1;

                    A++;

                }
                else { ; }

                //



            }
            //            printf("success02\n");
        }
        //printf("success03\n");
    }

    //    printf("success04\n");
    //    i = 0; j = Nbins*0.5; k = Nbins*0.5; l = Nbins*0.5;
    printf("%d,%d,%d\n", A, B, C);
    //    printf("%f,%f,%f\n", *(T + i * Nbins * Nbins * Nbins + j * Nbins * Nbins + k * Nbins + l-1));

    return (T);
}

void write_model(short int* T, std::string directory, int Nbins,
    float length_voxel, int FLAG, int(*p)[3]) {
    int i, j, k;
    int flag;
    //int Nbins = Nb;
    std::ofstream fout(directory, std::ios::binary);
    fout.write((char*)&(Nbins), sizeof(int));
    fout.write((char*)&(length_voxel), sizeof(float));
    for (i = 0; i < Nbins;i++) {
        for (j = 0;j < Nbins;j++) {
            for (k = 0;k < Nbins;k++) {

                fout.write((char*)&(*(T + i * Nbins * Nbins +
                    j * Nbins + k)), sizeof(short int));

            }
        }
    }
    fout.write((char*)&(FLAG), sizeof(int));
    for (flag = 0;flag < FLAG;flag++) {
        fout.write((char*)&(p[flag][0]), sizeof(int));
        fout.write((char*)&(p[flag][1]), sizeof(int));
        fout.write((char*)&(p[flag][2]), sizeof(int));
    }
    fout.close();
    std::cout << "model written\n";
}

short int* read_model() {
    int i, j, k;
    int Nbins;
    short int* T;
    std::ifstream fin("model.dat", std::ios::binary);
    fin.read((char*)&Nbins, sizeof(int));
    T = (short int*)malloc(1 * pow(Nbins, 3) * sizeof(short int));
    for (i = 0; i < Nbins;i++) {
        for (j = 0;j < Nbins;j++) {
            for (k = 0;k < Nbins;k++) {

                fin.read((char*)&(*(T + i * Nbins * Nbins +
                    j * Nbins + k)), sizeof(short int));

            }
        }
    }
    fin.close();
    return (T);
    std::cout << "model written\n";
}

std::tuple<int, int(*)[3] > find_vertex(int nbins, short int* t) {
    int i, j, k;
    static int vertex[N][3], flag = 0;
    int(*p)[3];
    for (i = 0;i < nbins-1;i++) {
        for (j = 0;j < nbins-1;j++) {
            for (k = 0;k < nbins - 1;k++) {
                if (*(t + i * nbins * nbins + j * nbins + k) != *(t + (i+1) * nbins * nbins + (j+1) * nbins + k + 1))
                {
                    if (std::min_element(t + i * nbins * nbins + j * nbins + k,
                        t + (i+1) * nbins * nbins + (j+1) * nbins + k + 1) != t + i * nbins * nbins + j * nbins + k)
                    {
                        vertex[flag][0] = i;
                        vertex[flag][1] = j;
                        vertex[flag][2] = k;
                    }
                    else
                    {
                        vertex[flag][0] = i;
                        vertex[flag][1] = j;
                        vertex[flag][2] = k + 1;

                    }
                    flag++;
                }



            }
        }
    }

    p = vertex;
    return std::make_tuple(flag, p);
}

void plot_geo(int FLAG, int vertex[N][3]) {
    using namespace matplot;
    std::vector<double> x, y, z;
    int flag;
    x.reserve(FLAG * sizeof(double));
    y.reserve(FLAG * sizeof(double));
    z.reserve(FLAG * sizeof(double));
    for (flag = 0;flag < FLAG;flag++) {
        x.push_back(vertex[flag][0]);
        y.push_back(vertex[flag][1]);
        z.push_back(vertex[flag][2]);
    }

    scatter3(x, y, z);
    show();

}

int main()
{
    int Nbins = Nb;
    float length_voxel = 0.1e-3;
    int FLAG;
    int(*p)[3];
    int mag = 40;
    char directory[] = "C:\\Users\\Administrator\\source\\data\\model_slab.dat";
    short int* T = create3Dmodel(Nbins,length_voxel);
    short int* T_lr = create3Dmodel_lr(int(Nbins / mag), mag,length_voxel);
    std::tie(FLAG, p) = find_vertex(int(Nbins / mag), T_lr);
    write_model(T, directory, Nbins, length_voxel, FLAG, p);
    plot_geo(FLAG, p);
}

