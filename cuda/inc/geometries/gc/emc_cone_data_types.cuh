#include <stddef.h>

//    Essa struct deve ser o suficiente para que os raios atinjam todos os 
//  voxels do bloco de recon como se o volume inteiro estivesse dentro da 
//  GPU.
//  Além disso, o número de camadas de reconstrução (NZ) dentro da GPU deve 
//  satisfazer NZ = nz_upper_width + lab.nz + nz_lower_width.
struct BulkBoundary {
    int nz_upper_width = 0; // higher z (smaller nz).
    int nz_lower_width = 0; // lower  z (higher  nz). 
    bool outter_boundary = true;
};

struct Lab {
    float Lx, Ly, Lz; // phantom size in each direction.
    float x0, y0, z0;
    int nx, ny, nz; // number of phantom points in each direction.
    float sx, sy, sz; // x-ray source position (x, y, z).
    int nbeta; // number of angles.
    int ndetc; // number of detector pixels.
    int n_ray_points; // number of integration points in the x-ray path.
};

/* to be implemented in the next version. 
       less args to pass on to functions. 
       kernel args together and encapsulated.
struct SuperLab {
    float Lx, Ly, Lz;   // phantom size in each direction.
    float x0, y0, z0;   // phantom central position.
    int nx, ny, nz;     // number of phantom points in each direction.
    float sx, sy, sz;   // x-ray source position (x, y, z).
    float *angs = NULL; // angles (radians).
    int nangs;          // number of angles.
    float *px = NULL, *py = NULL, *pz = NULL;           // position of detector pixels.
    float *flat = NULL, *dark = NULL, *flat_var = NULL; // flat, dark: mean value; flat: variance.
    int ndetc;                                          // number of detector pixels.
    int n_ray_points;    // number of integration points (x-ray path).
    struct BulkBoundary bounds;
};
*/