#ifndef _COLLIDE_H_
#define _COLLIDE_H_

// define const
#define STRIDE 3

// Structure declaration
typedef struct {
    int num_faces;
    float* vtx;
} Mesh;

typedef struct {
    float x;
    float y;
    float z;
} Vector;

typedef struct {
    int sphere;
    int face;
} CollidePair;

struct ResultCmp {
    __host__ __device__
    bool operator()(const CollidePair& o1, const CollidePair& o2) {
        if (o1.sphere == o2.sphere) {
            return o1.face < o2.face;
        }
        return o1.sphere < o2.sphere;
    }
};

#endif
