#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "collide.h"

#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include "tinyobj_loader_c.h"

static const char* mmap_file(size_t* len, const char* filename) {
    FILE* f;
    long file_size;
    struct stat sb;
    char* p;
    int fd;

    (*len) = 0;

    f = fopen(filename, "r");
    fseek(f, 0, SEEK_END);
    file_size = ftell(f);
    fclose(f);

    fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("open");
        return NULL;
    }

    if (fstat(fd, &sb) == -1) {
        perror("fstat");
        return NULL;
    }

    if (!S_ISREG(sb.st_mode)) {
        fprintf(stderr, "%s is not a file\n", "lineitem.tbl");
        return NULL;
    }

    p = (char*)mmap(0, (size_t)file_size, PROT_READ, MAP_SHARED, fd, 0);

    if (p == MAP_FAILED) {
        perror("mmap");
        return NULL;
    }

    if (close(fd) == -1) {
        perror("close");
        return NULL;
    }

    (*len) = (size_t)file_size;

    return p;
}

static const char* get_file_data(size_t* len, const char* filename) {
    const char* ext = strrchr(filename, '.');

    size_t data_len = 0;
    const char* data = NULL;

    if (strcmp(ext, ".gz") == 0) {
        assert(0);

    } else {
        data = mmap_file(&data_len, filename);
    }

    (*len) = data_len;
    return data;
}

static int LoadObjAndConvert(float bmin[3], float bmax[3], Mesh *mesh,
                      const char* filename) {
    tinyobj_attrib_t attrib;
    tinyobj_shape_t* shapes = NULL;
    size_t num_shapes;
    tinyobj_material_t* materials = NULL;
    size_t num_materials;

    size_t data_len = 0;
    const char* data = get_file_data(&data_len, filename);
    if (data == NULL) {
        exit(-1);
    }

    {
        unsigned int flags = TINYOBJ_FLAG_TRIANGULATE;
        int ret = tinyobj_parse_obj(&attrib, &shapes, &num_shapes, &materials,
                                    &num_materials, data, data_len, flags);
        if (ret != TINYOBJ_SUCCESS) {
            return -1;
        }
    }

    bmin[0] = bmin[1] = bmin[2] = FLT_MAX;
    bmax[0] = bmax[1] = bmax[2] = -FLT_MAX;

    {
        size_t face_offset = 0;
        size_t i;

        /* Assume triangulated face. */
        size_t num_triangles = attrib.num_face_num_verts;
        size_t stride = 3;

        mesh->num_faces = num_triangles;
        mesh->vtx = (float*)malloc(sizeof(float) * stride * num_triangles * 3);
        float* vb = mesh->vtx;

        for (i = 0; i < attrib.num_face_num_verts; i++) {
            size_t f;
            assert(attrib.face_num_verts[i] % 3 ==0); /* assume all triangle faces. */
            for (f = 0; f < (size_t)attrib.face_num_verts[i] / 3; f++) {
                size_t k;
                float v[3][3];

                tinyobj_vertex_index_t idx0 = attrib.faces[face_offset + 3 * f + 0];
                tinyobj_vertex_index_t idx1 = attrib.faces[face_offset + 3 * f + 1];
                tinyobj_vertex_index_t idx2 = attrib.faces[face_offset + 3 * f + 2];
                int f0 = idx0.v_idx;
                int f1 = idx1.v_idx;
                int f2 = idx2.v_idx;
                assert(f0 >= 0);
                assert(f1 >= 0);
                assert(f2 >= 0);

                for (k = 0; k < 3; k++) {
                    v[0][k] = attrib.vertices[3 * (size_t)f0 + k];
                    v[1][k] = attrib.vertices[3 * (size_t)f1 + k];
                    v[2][k] = attrib.vertices[3 * (size_t)f2 + k];
                    bmin[k] = (v[0][k] < bmin[k]) ? v[0][k] : bmin[k];
                    bmin[k] = (v[1][k] < bmin[k]) ? v[1][k] : bmin[k];
                    bmin[k] = (v[2][k] < bmin[k]) ? v[2][k] : bmin[k];
                    bmax[k] = (v[0][k] > bmax[k]) ? v[0][k] : bmax[k];
                    bmax[k] = (v[1][k] > bmax[k]) ? v[1][k] : bmax[k];
                    bmax[k] = (v[2][k] > bmax[k]) ? v[2][k] : bmax[k];
                }

                for (k = 0; k < 3; k++) {
                    vb[(3 * i + k) * stride + 0] = v[k][0];
                    vb[(3 * i + k) * stride + 1] = v[k][1];
                    vb[(3 * i + k) * stride + 2] = v[k][2];
                }
            }
            face_offset += (size_t)attrib.face_num_verts[i];
        }

    }

    tinyobj_attrib_free(&attrib);
    tinyobj_shapes_free(shapes, num_shapes);
    tinyobj_materials_free(materials, num_materials);

    return 0;
}

__device__ float MAX(float a, float b) {
    return (a>b)?(a):(b);
}

__device__ float MIN(float a, float b) {
    return (a<b)?(a):(b);
}

__global__ void cuda_stage1mesh(float *face, int *subdiv_face, int *face_min_bin, int *face_max_bin,
                                float *bmin, float binsize, int facenum) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= facenum) return;
    subdiv_face[id] = 1;
    for (int k = 0; k < 3; k++) {
        float f0 = face[(3 * id + 0) * STRIDE + k];
        float f1 = face[(3 * id + 1) * STRIDE + k];
        float f2 = face[(3 * id + 2) * STRIDE + k];
        float smax = MAX(MAX(f0, f1), f2);
        float smin = MIN(MIN(f0, f1), f2);

        face_min_bin[id * 3 + k] = floorf(smin / binsize) - floorf(bmin[k] / binsize);
        face_max_bin[id * 3 + k] = ceilf(smax / binsize) - floorf(bmin[k] / binsize);
        int num = face_max_bin[id * 3 + k] - face_min_bin[id * 3 + k];
        subdiv_face[id] *= num;
    }
}

__global__ void cuda_stage1sphere(Vector *sphere, int *subdiv_sphere, int *sphere_min_bin, int *sphere_max_bin,
                                float *bmin, float binsize, float radius, int spherenum) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= spherenum) return;
    sphere_max_bin[id * 3 + 0] = ceilf((sphere[id].x + radius) / binsize)
                                    - floorf(bmin[0] / binsize);
    sphere_min_bin[id * 3 + 0] = floorf((sphere[id].x - radius) / binsize)
                                    - floorf(bmin[0] / binsize);
    sphere_max_bin[id * 3 + 1] = ceilf((sphere[id].y + radius) / binsize)
                                    - floorf(bmin[1] / binsize);
    sphere_min_bin[id * 3 + 1] = floorf((sphere[id].y - radius) / binsize)
                                    - floorf(bmin[1] / binsize);
    sphere_max_bin[id * 3 + 2] = ceilf((sphere[id].z + radius) / binsize)
                                    - floorf(bmin[2] / binsize);
    sphere_min_bin[id * 3 + 2] = floorf((sphere[id].z - radius) / binsize)
                                    - floorf(bmin[2] / binsize);

    int xnum = sphere_max_bin[id * 3 + 0] - sphere_min_bin[id * 3 + 0];
    int ynum = sphere_max_bin[id * 3 + 1] - sphere_min_bin[id * 3 + 1];
    int znum = sphere_max_bin[id * 3 + 2] - sphere_min_bin[id * 3 + 2];
    subdiv_sphere[id] = xnum * ynum * znum;
}

__global__ void cuda_stage3mesh(int *subdiv_face, int *face_min_bin, int *face_max_bin,
                                    int *face_value, int *face_key, int binnum0, int binnum1, int facenum) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= facenum) return;
    int n;
    if (id == 0) { n = 0; }
        else { n = subdiv_face[id-1];}
    for (int k = face_min_bin[id * 3 + 2]; k < face_max_bin[id * 3 + 2]; k++)
        for (int j = face_min_bin[id * 3 + 1]; j < face_max_bin[id * 3 + 1]; j++)
            for (int i = face_min_bin[id * 3 + 0]; i < face_max_bin[id * 3 + 0]; i++) {
                face_value[n] = id;
                face_key[n] = i + j * binnum0 + k * binnum0 * binnum1;
                n++;
            }
}

__global__ void cuda_stage3sphere(int *subdiv_sphere, int *sphere_min_bin, int *sphere_max_bin,
                                    int *sphere_value, int *sphere_key, int binnum0, int binnum1, int spherenum) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= spherenum) return;
    int n;
    if (id == 0) { n = 0; }
    else { n = subdiv_sphere[id-1]; }
    for (int k = sphere_min_bin[id * 3 + 2]; k < sphere_max_bin[id * 3 + 2]; k++)
        for (int j = sphere_min_bin[id * 3 + 1]; j < sphere_max_bin[id * 3 + 1]; j++)
            for (int i = sphere_min_bin[id * 3 + 0]; i < sphere_max_bin[id * 3 + 0]; i++) {
                sphere_value[n] = id;
                sphere_key[n] = i + j * binnum0 + k * binnum0 * binnum1;
                n++;
            }
}

__global__ void cuda_stage5mesh(int *face_key, int *C_face, int bin_num, int facemapsize) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id < facemapsize && id > 0) {
        if (face_key[id] != face_key[id-1]){
            C_face[face_key[id]] = id;
        }
    } else if (id == 0) {
        C_face[face_key[id]] = id;
    }
}

__global__ void cuda_stage5sphere(int *sphere_key, int *C_sphere, int bin_num, int spheremapsize) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id < spheremapsize && id > 0) {
        if (sphere_key[id] != sphere_key[id-1]){
            C_sphere[sphere_key[id]] = id;
        }
    } else if (id == 0) {
        C_sphere[sphere_key[id]] = id;
    }
}


__device__ float dotproduct(Vector A, Vector B) {
    return A.x * B.x + A.y * B.y + A.z * B.z;
}

__device__ bool isIntersection(Vector A, Vector B, Vector C, float radius) {

    // Testing if sphere lies outside the triangle plane
    Vector V; // Normal
    V.x = ((B.y - A.y) * (C.z - A.z) - (B.z - A.z) * (C.y - A.y));
    V.y = ((B.z - A.z) * (C.x - A.x) - (B.x - A.x) * (C.z - A.z));
    V.z = ((B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x));
    float radiusSqr = radius*radius;
    float d = dotproduct(A, V);
    float e = dotproduct(V, V);
    if (d * d > e * radiusSqr) return 0;

    // Testing if sphere lies outside a triangle vertex
    float aa = dotproduct(A, A);
    float ab = dotproduct(A, B);
    float ac = dotproduct(A, C);
    float bb = dotproduct(B, B);
    float bc = dotproduct(B, C);
    float cc = dotproduct(C, C);
    if (aa > radiusSqr && ab > aa && ac > aa) return 0;
    if (bb > radiusSqr && ab > bb && bc > bb) return 0;
    if (cc > radiusSqr && ac > cc && bc > cc) return 0;

    // Testing if sphere lies outside a triangle edge`
    Vector AB, BC, CA, Q1, Q2, Q3, QC, QA, QB;
    AB.x = B.x - A.x; AB.y = B.y - A.y; AB.z = B.z - A.z;
    BC.x = C.x - B.x; BC.y = C.y - B.y; BC.z = C.z - B.z;
    CA.x = A.x - C.x; CA.y = A.y - C.y; CA.z = A.z - C.z;

    float d1 = dotproduct(A, AB);
    float e1 = dotproduct(AB, AB);
    float t1 = -d1/e1;
    if (t1 < 0.0f) t1 = 0.0f;
    if (t1 > 1.0f) t1 = 1.0f;
    Q1.x = A.x + t1 * AB.x; Q1.y = A.y + t1 * AB.y; Q1.z = A.z + t1 * AB.z;
    QC.x = C.x - Q1.x; QC.y = C.y - Q1.y; QC.z = C.z - Q1.z;
    if (dotproduct(Q1, Q1) > radiusSqr && dotproduct(Q1, QC) > 0) return 0;

    float d2 = dotproduct(B, BC);
    float e2 = dotproduct(BC, BC);
    float t2 = -d2/e2;
    if (t2 < 0.0f) t2 = 0.0f;
    if (t2 > 1.0f) t2 = 1.0f;
    Q2.x = B.x + t2 * BC.x; Q2.y = B.y + t2 * BC.y; Q2.z = B.z + t2 * BC.z;
    QA.x = A.x - Q2.x; QA.y = A.y - Q2.y; QA.z = A.z - Q2.z;
    if (dotproduct(Q2, Q2) > radiusSqr && dotproduct(Q2, QA) > 0) return 0;

    float d3 = dotproduct(C, CA);
    float e3 = dotproduct(CA, CA);
    float t3 = -d3/e3;
    if (t3 < 0.0f) t3 = 0.0f;
    if (t3 > 1.0f) t3 = 1.0f;
    Q3.x = C.x + t3 * CA.x; Q3.y = C.y + t3 * CA.y; Q3.z = C.z +t3 * CA.z;
    QB.x = B.x - Q3.x; QB.y = B.y - Q3.y; QB.z = B.z - Q3.z;
    if (dotproduct(Q3, Q3) > radiusSqr && dotproduct(Q3, QB) > 0) return 0;

    return 1;
}

__global__ void cuda_collide1(float *face, Vector *sphere, int* face_key, int* face_value,
                              int *sphere_key, int *sphere_value, int *C_face, int *C_sphere,
                              float radius, int bin_num, int facemapsize, int spheremapsize,
                              int *resultnum) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id < bin_num) {
        if (C_face[id] < 0 || C_sphere[id] < 0) return;
        for (int i = C_face[id]; i < facemapsize && id == face_key[i]; i++) {
            for (int j = C_sphere[id]; j < spheremapsize && id == sphere_key[j]; j++) {
                Vector A, B, C, S;
                S.x = sphere[sphere_value[j]].x;
                S.y = sphere[sphere_value[j]].y;
                S.z = sphere[sphere_value[j]].z;

                A.x = face[face_value[i] * 3 * STRIDE + 0] - S.x;
                A.y = face[face_value[i] * 3 * STRIDE + 1] - S.y;
                A.z = face[face_value[i] * 3 * STRIDE + 2] - S.z;
                B.x = face[face_value[i] * 3 * STRIDE + 3] - S.x;
                B.y = face[face_value[i] * 3 * STRIDE + 4] - S.y;
                B.z = face[face_value[i] * 3 * STRIDE + 5] - S.z;
                C.x = face[face_value[i] * 3 * STRIDE + 6] - S.x;
                C.y = face[face_value[i] * 3 * STRIDE + 7] - S.y;
                C.z = face[face_value[i] * 3 * STRIDE + 8] - S.z;

                if (isIntersection(A, B, C, radius))
                    resultnum[id]++;
            }
        }
    }
}

__global__ void cuda_collide2(float *face, Vector *sphere, int* face_key, int* face_value,
                              int *sphere_key, int *sphere_value, int *C_face, int *C_sphere,
                              float radius, int bin_num, int facemapsize, int spheremapsize,
                              CollidePair* result, int *resultnum, int *collide_num) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id < bin_num) {
        if (C_face[id] < 0 || C_sphere[id] < 0 || resultnum[id] == 0) return;
        for (int i = C_face[id]; i < facemapsize && id == face_key[i]; i++) {
            for (int j = C_sphere[id]; j < spheremapsize && id == sphere_key[j]; j++) {
                Vector A, B, C, S;
                S.x = sphere[sphere_value[j]].x;
                S.y = sphere[sphere_value[j]].y;
                S.z = sphere[sphere_value[j]].z;

                A.x = face[face_value[i] * 3 * STRIDE + 0] - S.x;
                A.y = face[face_value[i] * 3 * STRIDE + 1] - S.y;
                A.z = face[face_value[i] * 3 * STRIDE + 2] - S.z;
                B.x = face[face_value[i] * 3 * STRIDE + 3] - S.x;
                B.y = face[face_value[i] * 3 * STRIDE + 4] - S.y;
                B.z = face[face_value[i] * 3 * STRIDE + 5] - S.z;
                C.x = face[face_value[i] * 3 * STRIDE + 6] - S.x;
                C.y = face[face_value[i] * 3 * STRIDE + 7] - S.y;
                C.z = face[face_value[i] * 3 * STRIDE + 8] - S.z;

                if (isIntersection(A, B, C, radius)) {
                    int idx = atomicAdd(collide_num, 1);
                    result[idx].sphere = sphere_value[j];
                    result[idx].face = face_value[i];
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {

    // Set timers
    cudaEvent_t startEvent, stop1, stop2, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventRecord(startEvent, 0);
    printf("Start...\n");

    if (argc < 5) {
        fprintf(stderr, "Usage: ./collide meshfile spherefile radius outfile\n");
        exit(-1);
    }

    // Load & conv meshfile
    float bmin[3], bmax[3];
    Mesh mesh;
    mesh.vtx = NULL;
    if (LoadObjAndConvert(bmin, bmax, &mesh, argv[1]) < 0) {
        printf("failed to load & conv meshfile\n");
        exit(-1);
    }
    printf("Successfully load meshfile, facenum: %d\n", mesh.num_faces);

    // Load & conv spherefile
    FILE *fp = fopen(argv[2],"r");
    if (fp == NULL) {
        printf("failed to load & conv spherefile\n");
        exit(-1);
    }
    char* buffer = (char*)malloc(20 * sizeof(char));
    fgets(buffer, 20, fp);
    if (strcmp(buffer, "x,y,z\n") != 0) {
        printf("failed to load & conv spherefile\n");
        free(buffer);
        exit(-1);
    }
    free(buffer);

    float smin[3], smax[3];
    int num_sphere = 0;
    int size = 64;
    Vector *sphere = (Vector*)malloc(size * sizeof(Vector));
    smin[0] = smin[1] = smin[2] = FLT_MAX;
    smax[0] = smax[1] = smax[2] = -FLT_MAX;

    while (!feof(fp)) {
        fscanf(fp, "%f,%f,%f\n", &sphere[num_sphere].x, &sphere[num_sphere].y, &sphere[num_sphere].z);
        smin[0] = (sphere[num_sphere].x < smin[0]) ? sphere[num_sphere].x : smin[0];
        smin[1] = (sphere[num_sphere].y < smin[1]) ? sphere[num_sphere].y : smin[1];
        smin[2] = (sphere[num_sphere].z < smin[2]) ? sphere[num_sphere].z : smin[2];
        smax[0] = (sphere[num_sphere].x > smax[0]) ? sphere[num_sphere].x : smax[0];
        smax[1] = (sphere[num_sphere].y > smax[1]) ? sphere[num_sphere].y : smax[1];
        smax[2] = (sphere[num_sphere].z > smax[2]) ? sphere[num_sphere].z : smax[2];
        num_sphere++;
        if (num_sphere >= size) {
            size *= 2;
            sphere = (Vector*)realloc(sphere, size * sizeof(Vector));
        }
    }
    fclose(fp);
    printf("Successfully load spherefile, spherenum: %d\n", num_sphere);

    cudaEventCreate(&stop1);
    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);
    float time1;
    cudaEventElapsedTime(&time1, startEvent, stop1);
    printf("file loading time: %f ms\n", time1);
    printf("===================================================\n");

    // Exec
    // Stage 1
    // we set binsize = 2*radius
    float radius = atof(argv[3]);
    float binsize = 2*radius;
    printf("radius = %f, binsize = %f\n", radius, binsize);
    int binnum[3];
    for (int i = 0; i < 3; i++) {
        bmin[i] = (bmin[i] < smin[i] - radius) ? bmin[i] : smin[i] - radius;
        bmax[i] = (bmax[i] > smax[i] + radius) ? bmax[i] : smax[i] + radius;
        binnum[i] = ceilf(bmax[i] / binsize) - floorf(bmin[i] / binsize);
    }

    printf("xmin = %.3f, ymin = %.3f, zmin = %.3f\n", bmin[0], bmin[1], bmin[2]);
    printf("xmax = %.3f, ymax = %.3f, zmax = %.3f\n", bmax[0], bmax[1], bmax[2]);
    printf("xbin = %d, ybin = %d, zbin = %d\n", binnum[0], binnum[1], binnum[2]);
    printf("===================================================\n");

    // Start GPU computing
    float *d_min, *d_max;
    float *d_face;
    Vector *d_sphere;
    cudaMalloc(&d_min, sizeof(float) * 3);
    cudaMalloc(&d_max, sizeof(float) * 3);
    cudaMalloc(&d_face, sizeof(float) * mesh.num_faces * STRIDE * 3);
    cudaMalloc(&d_sphere, sizeof(Vector) * num_sphere);
    cudaMemcpy(d_min, bmin, sizeof(float) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, bmax, sizeof(float) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_face, mesh.vtx, sizeof(float) * mesh.num_faces * STRIDE * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sphere, sphere, sizeof(Vector) * num_sphere, cudaMemcpyHostToDevice);

    int *subdiv_face, *face_min_bin, *face_max_bin;
    cudaMalloc(&subdiv_face, sizeof(int) * mesh.num_faces);
    cudaMalloc(&face_min_bin, sizeof(int) * mesh.num_faces * 3);
    cudaMalloc(&face_max_bin, sizeof(int) * mesh.num_faces * 3);

    int *subdiv_sphere, *sphere_min_bin, *sphere_max_bin;
    cudaMalloc(&subdiv_sphere, sizeof(int) * num_sphere);
    cudaMalloc(&sphere_min_bin, sizeof(int) * num_sphere * 3);
    cudaMalloc(&sphere_max_bin, sizeof(int) * num_sphere * 3);

    cuda_stage1mesh<<<(mesh.num_faces+1023)/1024, 1024>>>(d_face, subdiv_face, face_min_bin, face_max_bin,
                                                            d_min, binsize, mesh.num_faces);
    cuda_stage1sphere<<<(num_sphere+1023)/1024, 1024>>>(d_sphere, subdiv_sphere, sphere_min_bin, sphere_max_bin,
                                                            d_min, binsize, radius, num_sphere);
    cudaDeviceSynchronize();
    cudaFree(d_min); cudaFree(d_max);

    // Stage 2
    thrust::device_ptr<int> t_subdiv_face(subdiv_face);
    thrust::device_ptr<int> t_subdiv_sphere(subdiv_sphere);
    thrust::inclusive_scan(t_subdiv_face, t_subdiv_face + mesh.num_faces, t_subdiv_face);
    thrust::inclusive_scan(t_subdiv_sphere, t_subdiv_sphere + num_sphere, t_subdiv_sphere);
    int FaceMapSize, SphereMapSize;
    cudaMemcpy(&FaceMapSize, subdiv_face + mesh.num_faces - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&SphereMapSize, subdiv_sphere + num_sphere - 1, sizeof(int), cudaMemcpyDeviceToHost);
    //printf("%d, %d\n", FaceMapSize, SphereMapSize);

    // Stage 3
    int *d_face_key, *d_face_value, *d_sphere_key, *d_sphere_value;
    cudaMalloc(&d_face_value, FaceMapSize * sizeof(int));
    cudaMalloc(&d_face_key, FaceMapSize * sizeof(int));
    cudaMalloc(&d_sphere_value, SphereMapSize * sizeof(int));
    cudaMalloc(&d_sphere_key, SphereMapSize * sizeof(int));

    cuda_stage3mesh<<<(mesh.num_faces+1023)/1024, 1024>>>(subdiv_face, face_min_bin, face_max_bin,
                                                            d_face_value, d_face_key, binnum[0], binnum[1],
                                                                mesh.num_faces);
    cuda_stage3sphere<<<(num_sphere+1023)/1024, 1024>>>(subdiv_sphere, sphere_min_bin, sphere_max_bin,
                                                            d_sphere_value, d_sphere_key, binnum[0], binnum[1],
                                                                num_sphere);
    cudaDeviceSynchronize();

    cudaFree(subdiv_face); cudaFree(face_min_bin); cudaFree(face_max_bin);
    cudaFree(subdiv_sphere); cudaFree(sphere_min_bin); cudaFree(sphere_max_bin);

    // Stage 4
    thrust::device_ptr<int> t_face_key(d_face_key);
    thrust::device_ptr<int> t_face_value(d_face_value);
    thrust::device_ptr<int> t_sphere_key(d_sphere_key);
    thrust::device_ptr<int> t_sphere_value(d_sphere_value);
    thrust::sort_by_key(t_face_key, t_face_key + FaceMapSize, t_face_value);
    thrust::sort_by_key(t_sphere_key, t_sphere_key + SphereMapSize, t_sphere_value);

    // Stage 5
    int bin_num = binnum[0] * binnum[1] * binnum[2];
    int *d_C_face, *d_C_sphere;
    cudaMalloc(&d_C_sphere, sizeof(int) * bin_num);
    cudaMalloc(&d_C_face, sizeof(int) * bin_num);
    cudaMemset(d_C_sphere, -1, sizeof(int) * bin_num);
    cudaMemset(d_C_face, -1, sizeof(int) * bin_num);

    cuda_stage5mesh<<<(FaceMapSize+1023)/1024, 1024>>>(d_face_key, d_C_face, bin_num, FaceMapSize);
    cuda_stage5sphere<<<(SphereMapSize+1023)/1024, 1024>>>(d_sphere_key, d_C_sphere, bin_num, SphereMapSize);
    cudaDeviceSynchronize();

    cudaEventCreate(&stop2);
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    float time2;
    cudaEventElapsedTime(&time2, stop1, stop2);

    // Skip stage 6

    // GPU collision detection Stage
    int *d_resultnum;
    int *d_collide_num;
    cudaMalloc(&d_collide_num, sizeof(int));
    cudaMalloc(&d_resultnum, sizeof(int) * bin_num);
    cudaMemset(d_collide_num, 0, sizeof(int));
    cudaMemset(d_resultnum, 0, sizeof(int) * bin_num);

    cuda_collide1<<<(bin_num+1023)/1024, 1024>>>(d_face, d_sphere, d_face_key, d_face_value,
                                                d_sphere_key, d_sphere_value, d_C_face, d_C_sphere,
                                                radius, bin_num, FaceMapSize, SphereMapSize,
                                                d_resultnum);
    cudaDeviceSynchronize();

    thrust::device_ptr<int> t_resultnum(d_resultnum);
    int collide_num = thrust::reduce(t_resultnum, t_resultnum + bin_num);

    CollidePair *d_result;
    cudaMalloc(&d_result, sizeof(CollidePair) * collide_num);
    cuda_collide2<<<(bin_num+1023)/1024, 1024>>>(d_face, d_sphere, d_face_key, d_face_value,
                                                d_sphere_key, d_sphere_value, d_C_face, d_C_sphere,
                                                radius, bin_num, FaceMapSize, SphereMapSize,
                                                d_result, d_resultnum, d_collide_num);
    cudaDeviceSynchronize();

    thrust::device_ptr<CollidePair> t_result(d_result);
    thrust::sort(t_result, t_result + collide_num, ResultCmp());
    CollidePair *result = (CollidePair*)malloc(sizeof(CollidePair) * collide_num);
    cudaMemcpy(result, d_result, sizeof(CollidePair) * collide_num, cudaMemcpyDeviceToHost);

    cudaEventCreate(&stopEvent);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float time3, elapsedTime;
    cudaEventElapsedTime(&time3, stop2, stopEvent);
    cudaEventElapsedTime(&elapsedTime, stop1, stopEvent);

    printf("Stage 1-5 time: %f ms\n", time2);
    printf("GPU collision detection time: %f ms\n", time3);
    printf("total time (w/out loading time): %f ms\n", elapsedTime);

    // Output & Free memory
    int pair_num = 0;
    fp = fopen(argv[4],"w");
    fprintf(fp, "%f\n", elapsedTime);
    for (int i = 0; i < collide_num; i++) {
        if (i > 0) {
            if (result[i].sphere == result[i-1].sphere && result[i].face == result[i-1].face)
                continue;
        }
        fprintf(fp,"%d,%d\n",result[i].sphere, result[i].face);
        pair_num++;
    }
    fclose(fp);
    printf("COLLIDE_PAIRS: %d\n", pair_num);
    printf("FINISH!\n");

    cudaFree(d_face);  cudaFree(d_sphere);
    cudaFree(d_face_key); cudaFree(d_face_value);
    cudaFree(d_sphere_key); cudaFree(d_sphere_value);
    cudaFree(d_C_face); cudaFree(d_C_sphere);
    cudaFree(d_collide_num); cudaFree(d_resultnum); cudaFree(d_result);

    free(mesh.vtx); free(sphere);
    free(result);
    return 0;
}

