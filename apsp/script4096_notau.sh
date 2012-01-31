#PBS -S /bin/bash
#PBS -q regular
#PBS -V
#PBS -l mppwidth=24576
#PBS -l mppnppn=4
#PBS -l mppdepth=6
#PBS -j eo
#PBS -o dcapsp.hopper.1024nodes.f.out
#PBS -e dcapsp.hopper.1024nodes.f.out
#PBS -l walltime=00:20:00
#PBS -A mp156

cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=6
aprun -n 4096 -N 4 -d 6 -ss ./test_dcapsp -pdim 64 -n 4096 -b1 8 -b2 8 -test 0 -crep 1
aprun -n 4096 -N 4 -d 6 -ss ./test_dcapsp -pdim 64 -n 8192 -b1 16 -b2 16 -test 0 -crep 1
aprun -n 4096 -N 4 -d 6 -ss ./test_dcapsp -pdim 64 -n 16384 -b1 16 -b2 16 -test 0 -crep 1
aprun -n 4096 -N 4 -d 6 -ss ./test_dcapsp -pdim 64 -n 32768 -b1 32 -b2 32 -test 0 -crep 1

aprun -n 4096 -N 4 -d 6 -ss ./test_dcapsp -pdim 32 -n 4096 -b1 8 -b2 8 -test 0 -crep 4
aprun -n 4096 -N 4 -d 6 -ss ./test_dcapsp -pdim 32 -n 8192 -b1 8 -b2 16 -test 0 -crep 4
aprun -n 4096 -N 4 -d 6 -ss ./test_dcapsp -pdim 32 -n 16384 -b1 16 -b2 16 -test 0 -crep 4
aprun -n 4096 -N 4 -d 6 -ss ./test_dcapsp -pdim 32 -n 32768 -b1 16 -b2 32 -test 0 -crep 4

aprun -n 4096 -N 4 -d 6 -ss ./test_dcapsp -pdim 16 -n 4096 -b1 8 -b2 16 -test 0 -crep 16
aprun -n 4096 -N 4 -d 6 -ss ./test_dcapsp -pdim 16 -n 8192 -b1 16 -b2 16 -test 0 -crep 16
aprun -n 4096 -N 4 -d 6 -ss ./test_dcapsp -pdim 16 -n 16384 -b1 16 -b2 32 -test 0 -crep 16


aprun -n 4096 -N 4 -d 6 -ss ./test_dcapsp -pdim 64 -n 65536 -b1 32 -b2 32 -test 0 -crep 1

