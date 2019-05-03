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
#include <time.h>

#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include "tinyobj_loader_c.h"

// define const
#define STRIDE 3
#define MAX(a,b) (a>b)?(a):(b)
#define MIN(a,b) (a<b)?(a):(b)

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

float dotproduct(Vector A, Vector B) {
    return A.x * B.x + A.y * B.y + A.z * B.z;
}

int isIntersection(Vector A, Vector B, Vector C, float radius) {

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


int cmpfunc (const void* p, const void* q) {
    int f1 = ((CollidePair *)p)->face;
    int f2 = ((CollidePair *)q)->face;
    int s1 = ((CollidePair *)p)->sphere;
    int s2 = ((CollidePair *)q)->sphere;
    if (s1 == s2) {
        return f1 > f2;
    }
    return s1 > s2;
}

int main(int argc, char* argv[]) {

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
    float radius = atof(argv[3]);
    int num_sphere = 0;
    int size = 64;
    Vector *sphere = (Vector*)malloc(size * sizeof(Vector));

    while (!feof(fp)) {
        fscanf(fp, "%f,%f,%f\n", &sphere[num_sphere].x, &sphere[num_sphere].y, &sphere[num_sphere].z);
        num_sphere++;
        if (num_sphere >= size) {
            size *= 2;
            sphere = (Vector*)realloc(sphere, size * sizeof(Vector));
        }
    }
    printf("Successfully load spherefile, spherenum: %d\n", num_sphere);

    time_t start, end;
    double elapsed;
    time(&start);
    // Exec
    int resultsize = 64;
    int idx = 0;
    CollidePair* result = (CollidePair*)malloc(size * sizeof(CollidePair));
    for (int i = 0; i < mesh.num_faces; i++) {
        for (int j = 0; j < num_sphere; j++) {
            Vector A, B, C, S;
            S.x = sphere[j].x;
            S.y = sphere[j].y;
            S.z = sphere[j].z;

            A.x = mesh.vtx[i * 3 * STRIDE + 0] - S.x;
            A.y = mesh.vtx[i * 3 * STRIDE + 1] - S.y;
            A.z = mesh.vtx[i * 3 * STRIDE + 2] - S.z;
            B.x = mesh.vtx[i * 3 * STRIDE + 3] - S.x;
            B.y = mesh.vtx[i * 3 * STRIDE + 4] - S.y;
            B.z = mesh.vtx[i * 3 * STRIDE + 5] - S.z;
            C.x = mesh.vtx[i * 3 * STRIDE + 6] - S.x;
            C.y = mesh.vtx[i * 3 * STRIDE + 7] - S.y;
            C.z = mesh.vtx[i * 3 * STRIDE + 8] - S.z;

            if (isIntersection(A, B, C, radius) == 1) {
                result[idx].sphere = j;
                result[idx].face = i;
                idx++;
                if (idx >= resultsize) {
                    resultsize *= 2;
                    result = (CollidePair*)realloc(result, resultsize * sizeof(Vector));
                }
            }
        }
    }
    qsort(result, idx, sizeof(CollidePair), cmpfunc);
    time(&end);
    elapsed = difftime(end, start);
    printf("TIME: %d s\n", (int)elapsed);

    // Output & Free memory
    int pair_num = 0;
    fp = fopen(argv[4],"w");
    for (int i = 0; i < idx; i++) {
        if (i > 0) {
            if (result[i].sphere == result[i-1].sphere && result[i].face == result[i-1].face)
                continue;
        }
        fprintf(fp,"%d,%d\n",result[i].sphere, result[i].face);
        pair_num++;
    }
    fclose(fp);
    printf("PAIRS: %d\n", pair_num);

    printf("FINISH!\n");
    free(mesh.vtx); free(sphere);
    free(result);
    return 0;
}

