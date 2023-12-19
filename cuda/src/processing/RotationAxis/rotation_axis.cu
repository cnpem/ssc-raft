// Authors: Giovanni Baraldi, Gilberto Martinez, Eduardo Miqueles
// Sinogram centering

#include "../../../inc/processing.h"

extern "C"{

    int getRotAxisOfsset(float *start_frame, float *end_frame, float *flat, float *dark, dim3 size, int rotation_axis_method)
    {
        int rotation_axis_offset = 0;
        int nrays                = size.x;
        int nangles              = size.y;

        switch (rotation_axis_method){
            case 0:
                /* Centersino */
            rotation_axis_offset = getCentersino(start_frame, end_frame, dark, flat, nrays, nangles);
                break;
            case 1:
                /* 360 correlation */
                break;
            default:
                printf("No enough angle information to find the rotation axis deviation automatically. \n");
                printf("Setting rotation axis deviation as ZERO. \n");
                break;
        }	

        return rotation_axis_offset;
    }

    int getCentersino(float* frame0, float* frame1, float* dark, float* flat, int sizex, int sizey)
    {
        return Centersino(frame0, frame1, dark, flat, sizex, sizey);
    }

}
