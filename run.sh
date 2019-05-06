#!/usr/bin/env bash
#SBATCH -p slurm_shortgpu
#SBATCH --job-name=finalproject
#SBATCH --output="finalproject.out"
#SBATCH --time=0-00:05:00
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR
#./collide testcase/diamond.obj testcase/diamond_sphere.csv 20.0 result.out
#./collide testcase/cube.obj testcase/cube_sphere.csv 0.4 result.out
./collide testcase/sample_mesh.obj testcase/sample_spheres.csv 3.0 result.out

