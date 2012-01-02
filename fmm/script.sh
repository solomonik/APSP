#PBS -S /bin/bash
#PBS -V
#PBS -l mppwidth=1
#PBS -l mppnppn=1
#PBS -j eo
#PBS -o mm.6.out
#PBS -e mm.6.out
#PBS -A mp156

cd $PBS_O_WORKDIR
for file in execs/bench*
do
  aprun -n 1 -N 1 ./$file -n 128 -m 128 -k 128
  aprun -n 1 -N 1 ./$file -n 400 -m 400 -k 400
done
