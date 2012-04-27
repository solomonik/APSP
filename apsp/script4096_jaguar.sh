#PBS -S /bin/bash
#PBS -q regular
#PBS -V
#PBS -l mppwidth=24576
#PBS -j eo
#PBS -o dcapsp.jaguar.4096nodes.g.out
#PBS -e dcapsp.jaguar.4096nodes.g.out
#PBS -l walltime=00:20:00
#PBS -A mp156

cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=8
aprun -n 4096 -N 2 -d 8 -ss ./test_dcapsp -pdim 64 -n 4096 -b1 8 -b2 8 -test 0 -crep 1
aprun -n 4096 -N 2 -d 8 -ss ./test_dcapsp -pdim 64 -n 8192 -b1 16 -b2 16 -test 0 -crep 1
aprun -n 4096 -N 2 -d 8 -ss ./test_dcapsp -pdim 64 -n 16384 -b1 16 -b2 16 -test 0 -crep 1
aprun -n 4096 -N 2 -d 8 -ss ./test_dcapsp -pdim 64 -n 32768 -b1 32 -b2 32 -test 0 -crep 1

aprun -n 4096 -N 2 -d 8 -ss ./test_dcapsp -pdim 32 -n 4096 -b1 8 -b2 8 -test 0 -crep 4
aprun -n 4096 -N 2 -d 8 -ss ./test_dcapsp -pdim 32 -n 8192 -b1 8 -b2 16 -test 0 -crep 4
aprun -n 4096 -N 2 -d 8 -ss ./test_dcapsp -pdim 32 -n 16384 -b1 16 -b2 16 -test 0 -crep 4
aprun -n 4096 -N 2 -d 8 -ss ./test_dcapsp -pdim 32 -n 32768 -b1 16 -b2 32 -test 0 -crep 4

aprun -n 4096 -N 2 -d 8 -ss ./test_dcapsp -pdim 16 -n 4096 -b1 8 -b2 16 -test 0 -crep 16
aprun -n 4096 -N 2 -d 8 -ss ./test_dcapsp -pdim 16 -n 8192 -b1 16 -b2 16 -test 0 -crep 16
aprun -n 4096 -N 2 -d 8 -ss ./test_dcapsp -pdim 16 -n 16384 -b1 16 -b2 32 -test 0 -crep 16
aprun -n 4096 -N 2 -d 8 -ss ./test_dcapsp -pdim 16 -n 32768 -b1 32 -b2 32 -test 0 -crep 16


aprun -n 4096 -N 2 -d 8 -ss ./test_dcapsp -pdim 64 -n 65536 -b1 32 -b2 32 -test 0 -crep 1
aprun -n 4096 -N 2 -d 8 -ss ./test_dcapsp -pdim 32 -n 65536 -b1 32 -b2 32 -test 0 -crep 4
aprun -n 4096 -N 2 -d 8 -ss ./test_dcapsp -pdim 64 -n 131072 -b1 64 -b2 64 -test 0 -crep 1

