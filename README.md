# ME759 Final Project: Collision Detection

This project mainly applies GPU in parallel computing, using C to leverage CUDA and Thrust to accomplish the default final project 2: Collision Detection. The goal is to detecting collisions between a triangle mesh and a large collection of spheres. It implements a board-phase collision detection step in order to separate spheres and triangles into bins, then calculates all sphere-triangle pair with in each bins, applies brute-force approach at bin level only to perform collision detection parallelly, finally achieves a better performance.

## Building

This program should build on Euler with following commands 

```
> module load gcc cuda cmake
> cmake .
> make 
```

## Running/Input/Output:

To run the GPU parallel version:

```
./collide meshfile spherefile radius outfile 
```

To run the sequential brute-force version:

```
./collide_bf meshfile spherefile radius outfile 
```

* meshfile is the name of the input file that contains the mesh stored in the Wavefront OBJ format
  
* spherefile is the name of the CSV input file for all spheres

* radius is the radius of the spheres in the problem

* outfile is the name of the file that your program should generate to report all collisions

## Progress

* Successfully loaded & converted meshfile and spherefile

* Successfully implemented algorithm for sphere-triangle collision detection, accomplished a brute-force solution

* Successfully implemented Mazhar’s algorithm, a broad-phase collision detection step in order to spatially group spheres and triangles into bins

* Created & passed several simple testcase

* Passed official testcase

## Future work

* Optimization

* Accommodating more common situation

## Authors

* **Yifan Hong** - [yhong84@wisc.edu](yhong84@wisc.edu)