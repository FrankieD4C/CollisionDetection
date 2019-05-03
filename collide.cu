#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
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
        assert(0); /* todo */

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
        size_t stride = 3; // 3 = pos(3float)

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

float MAX(float a, float b) {
    return (a>b)?(a):(b);
}

float MIN(float a, float b) {
    return (a<b)?(a):(b);
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

    // Testing if sphere lies outside a triangle edge
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
                            float radius, int bin_num, int facesize, int spheresize,
                            int *resultnum) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id < bin_num) {
        if (C_face[id] < 0 || C_sphere[id] < 0) return;
        for (int i = C_face[id]; i < facesize && id == face_key[i]; i++) {
            for (int j = C_sphere[id]; j < spheresize && id == sphere_key[j]; j++) {
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
                             float radius, int bin_num, int facesize, int spheresize,
                             CollidePair* result, int *resultnum, int *collide_num) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id < bin_num) {
        if (C_face[id] < 0 || C_sphere[id] < 0 || resultnum[id] == 0) return;
        for (int i = C_face[id]; i < facesize && id == face_key[i]; i++) {
            for (int j = C_sphere[id]; j < spheresize && id == sphere_key[j]; j++) {
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

    cudaEvent_t startEvent, stop1, stop2, stop3, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventRecord(startEvent, 0);

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

    // Set timers
    cudaEventCreate(&stop1);
    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);
    float time1;
    cudaEventElapsedTime(&time1, startEvent, stop1);
    printf("file loading time: %f ms\n", time1);
    printf("===================================================\n");

    // Exec
    // Stage 1 (CAN BE PARALLELIZED)
    // we set binsize = 2*radius
    float radius = atof(argv[3]);
    printf("radius = %f\n", radius);
    float binsize = 2*radius;
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

    int* subdiv_face = (int*)malloc(mesh.num_faces * sizeof(int));
    int* face_min_bin = (int*)malloc(3 * mesh.num_faces * sizeof(int));
    int* face_max_bin = (int*)malloc(3 * mesh.num_faces * sizeof(int));
    for (int i = 0; i < mesh.num_faces; i++) {
        subdiv_face[i] = 1;
        for (int k = 0; k < 3; k++) {
            float f0 = mesh.vtx[(3 * i + 0) * STRIDE + k];
            float f1 = mesh.vtx[(3 * i + 1) * STRIDE + k];
            float f2 = mesh.vtx[(3 * i + 2) * STRIDE + k];
            float smax = MAX(MAX(f0, f1), f2);
            float smin = MIN(MIN(f0, f1), f2);

            face_min_bin[i * 3 + k] = floorf(smin / binsize) - floorf(bmin[k] / binsize);
            face_max_bin[i * 3 + k] = ceilf(smax / binsize) - floorf(bmin[k] / binsize);
            int num = face_max_bin[i * 3 + k] - face_min_bin[i * 3 + k];
            subdiv_face[i] *= num;
        }
        //printf("%d %d %d\n", face_max_bin[i * 3 + 0], face_max_bin[i * 3 + 1], face_max_bin[i * 3 + 2]);
        //printf("%d %d %d\n", face_min_bin[i * 3 + 0], face_min_bin[i * 3 + 1], face_min_bin[i * 3 + 2]);
        //printf("%d\n", subdiv_face[i]);
    }

    int* subdiv_sphere = (int*)malloc(num_sphere * sizeof(int));
    int* sphere_min_bin = (int*)malloc(3 * num_sphere * sizeof(int));
    int* sphere_max_bin = (int*)malloc(3 * num_sphere * sizeof(int));
    for (int i = 0; i < num_sphere; i++) {
        sphere_max_bin[i * 3 + 0] = ceilf((sphere[i].x + radius) / binsize)
                                        - floorf(bmin[0] / binsize);
        sphere_min_bin[i * 3 + 0] = floorf((sphere[i].x - radius) / binsize)
                                        - floorf(bmin[0] / binsize);
        sphere_max_bin[i * 3 + 1] = ceilf((sphere[i].y + radius) / binsize)
                                        - floorf(bmin[1] / binsize);
        sphere_min_bin[i * 3 + 1] = floorf((sphere[i].y - radius) / binsize)
                                        - floorf(bmin[1] / binsize);
        sphere_max_bin[i * 3 + 2] = ceilf((sphere[i].z + radius) / binsize)
                                        - floorf(bmin[2] / binsize);
        sphere_min_bin[i * 3 + 2] = floorf((sphere[i].z - radius) / binsize)
                                        - floorf(bmin[2] / binsize);

        int xnum = sphere_max_bin[i * 3 + 0] - sphere_min_bin[i * 3 + 0];
        int ynum = sphere_max_bin[i * 3 + 1] - sphere_min_bin[i * 3 + 1];
        int znum = sphere_max_bin[i * 3 + 2] - sphere_min_bin[i * 3 + 2];
        subdiv_sphere[i] = xnum * ynum * znum;
        //printf("%d %d %d %d\n", xnum, ynum, znum, subdiv_sphere[i]);
        //printf("%d ", subdiv_sphere[i]);
    }

    // Stage 2
    thrust::inclusive_scan(subdiv_face, subdiv_face + mesh.num_faces, subdiv_face);
    thrust::inclusive_scan(subdiv_sphere, subdiv_sphere + num_sphere, subdiv_sphere);
    //printf("%d, %d\n", subdiv_face[mesh.num_faces-1], subdiv_sphere[num_sphere-1]);

    // Stage 3
    int BFaceSize = subdiv_face[mesh.num_faces-1];
    int BSphereSize = subdiv_sphere[num_sphere-1];
    int *face_value = (int*)malloc(BFaceSize * sizeof(int));
    int *face_key = (int*)malloc(BFaceSize * sizeof(int));
    int *sphere_value = (int*)malloc(BSphereSize * sizeof(int));
    int *sphere_key = (int*)malloc(BSphereSize * sizeof(int));
    int count = 0;
    for (int n = 0; n < mesh.num_faces; n++) {
        for (int k = face_min_bin[n * 3 + 2]; k < face_max_bin[n * 3 + 2]; k++)
            for (int j = face_min_bin[n * 3 + 1]; j < face_max_bin[n * 3 + 1]; j++)
                for (int i = face_min_bin[n * 3 + 0]; i < face_max_bin[n * 3 + 0]; i++) {
                    face_value[count] = n;
                    face_key[count] = i + j * binnum[0] + k * binnum[0] * binnum[1];
                    count ++;
                }
    }
    count = 0;
    for (int n = 0; n < num_sphere; n++) {
        for (int k = sphere_min_bin[n * 3 + 2]; k < sphere_max_bin[n * 3 + 2]; k++)
            for (int j = sphere_min_bin[n * 3 + 1]; j < sphere_max_bin[n * 3 + 1]; j++)
                for (int i = sphere_min_bin[n * 3 + 0]; i < sphere_max_bin[n * 3 + 0]; i++) {
                    sphere_value[count] = n;
                    sphere_key[count] = i + j * binnum[0] + k * binnum[0] * binnum[1];
                    count ++;
                }
    }

    free(subdiv_face); free(face_min_bin); free(face_max_bin);
    free(subdiv_sphere); free(sphere_min_bin); free(sphere_max_bin);

    cudaEventCreate(&stop2);
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    float time2;
    cudaEventElapsedTime(&time2, stop1, stop2);

    // Stage 4
    int *d_face_key, *d_face_value, *d_sphere_key, *d_sphere_value;
    thrust::device_vector<int> t_face_key(face_key, face_key + BFaceSize);
    thrust::device_vector<int> t_face_value(face_value, face_value + BFaceSize);
    thrust::device_vector<int> t_sphere_key(sphere_key, sphere_key + BSphereSize);
    thrust::device_vector<int> t_sphere_value(sphere_value, sphere_value + BSphereSize);
    thrust::sort_by_key(t_face_key.begin(), t_face_key.end(), t_face_value.begin());
    thrust::sort_by_key(t_sphere_key.begin(), t_sphere_key.end(), t_sphere_value.begin());
    d_face_key = thrust::raw_pointer_cast(&t_face_key[0]);
    d_face_value = thrust::raw_pointer_cast(&t_face_value[0]);
    d_sphere_key = thrust::raw_pointer_cast(&t_sphere_key[0]);
    d_sphere_value = thrust::raw_pointer_cast(&t_sphere_value[0]);
    cudaMemcpy(face_key, d_face_key, sizeof(int) * BFaceSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(face_value, d_face_value, sizeof(int) * BFaceSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(sphere_key, d_sphere_key, sizeof(int) * BSphereSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(sphere_value, d_sphere_value, sizeof(int) * BSphereSize, cudaMemcpyDeviceToHost);

    // Stage 5
    int bin_num = binnum[0] * binnum[1] * binnum[2];
    int *C_face = (int*)malloc(bin_num * sizeof(int));
    int *C_sphere = (int*)malloc(bin_num * sizeof(int));
    int face_ptr = 0;
    int sphere_ptr = 0;
    for (int i = 0; i < bin_num; i++) {
        C_face[i] = face_ptr;
        int isActive = 0;
        while (face_ptr < BFaceSize && face_key[face_ptr] == i) {
            face_ptr++;
            isActive = 1;
        }
        if (isActive == 0) C_face[i] = -1;
        C_sphere[i] = sphere_ptr;
        isActive = 0;
        while (sphere_ptr < BSphereSize && sphere_key[sphere_ptr] == i) {
            sphere_ptr++;
            isActive = 1;
        }
        if (isActive == 0) C_sphere[i] = -1;
    }
    free(face_value); free(face_key);
    free(sphere_value); free(sphere_key);

    cudaEventCreate(&stop3);
    cudaEventRecord(stop3, 0);
    cudaEventSynchronize(stop3);
    float time3;
    cudaEventElapsedTime(&time3, stop2, stop3);

    // Temporarily skip stage 6

    // Stage 7
    float *d_face;
    Vector *d_sphere;
    int *d_C_face, *d_C_sphere;

    int *d_resultnum;
    int *d_collide_num;

    cudaMalloc(&d_face, sizeof(float) * mesh.num_faces * STRIDE * 3);
    cudaMalloc(&d_sphere, sizeof(Vector) * num_sphere);
    cudaMalloc(&d_C_sphere, sizeof(int) * bin_num);
    cudaMalloc(&d_C_face, sizeof(int) * bin_num);
    cudaMalloc(&d_collide_num, sizeof(int));
    cudaMalloc(&d_resultnum, sizeof(int) * bin_num);

    cudaMemcpy(d_face, mesh.vtx, sizeof(float) * mesh.num_faces * STRIDE * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sphere, sphere, sizeof(Vector) * num_sphere, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_sphere, C_sphere, sizeof(int) * bin_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_face, C_face, sizeof(int) * bin_num, cudaMemcpyHostToDevice);
    cudaMemset(d_collide_num, 0, sizeof(int));
    cudaMemset(d_resultnum, 0, sizeof(int) * bin_num);

    cuda_collide1<<<(bin_num+1023)/1024, 1024>>>(d_face, d_sphere, d_face_key, d_face_value,
                                                d_sphere_key, d_sphere_value, d_C_face, d_C_sphere,
                                                radius, bin_num, BFaceSize, BSphereSize,
                                                d_resultnum);
    cudaDeviceSynchronize();

    int* resultnum = (int*)malloc(sizeof(int) * bin_num);
    cudaMemcpy(resultnum, d_resultnum, sizeof(int) * bin_num, cudaMemcpyDeviceToHost);
    int collide_num = thrust::reduce(resultnum, resultnum + bin_num);

    CollidePair *d_result;
    cudaMalloc(&d_result, sizeof(CollidePair) * collide_num);
    cuda_collide2<<<(bin_num+1023)/1024, 1024>>>(d_face, d_sphere, d_face_key, d_face_value,
            d_sphere_key, d_sphere_value, d_C_face, d_C_sphere,
            radius, bin_num, BFaceSize, BSphereSize,
            d_result, d_resultnum, d_collide_num);
    cudaDeviceSynchronize();


    CollidePair *result = (CollidePair*)malloc(sizeof(CollidePair) * collide_num);
    cudaMemcpy(result, d_result, sizeof(CollidePair) * collide_num, cudaMemcpyDeviceToHost);
    thrust::sort(result, result + collide_num, ResultCmp());

    cudaEventCreate(&stopEvent);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float time4, elapsedTime;
    cudaEventElapsedTime(&time4, stop3, stopEvent);
    cudaEventElapsedTime(&elapsedTime, stop1, stopEvent);

    printf("Stage 1-3 time: %f ms\n", time2);
    printf("Stage 4-5 time: %f ms\n", time3);
    printf("GPU collision detection time: %f ms\n", time4);
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
    //printf("1ST_PAIRS: %d\n", collide_num);
    printf("COLLIDE_PAIRS: %d\n", pair_num);
    printf("FINISH!\n");

    cudaFree(d_face);  cudaFree(d_sphere);
    cudaFree(d_C_face); cudaFree(d_C_sphere);
    cudaFree(d_collide_num);
    cudaFree(d_resultnum); cudaFree(d_result);

    free(mesh.vtx); free(sphere);
    free(C_face); free(C_sphere);
    free(resultnum); free(result);
    return 0;
}

