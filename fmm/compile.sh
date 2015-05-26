#!/bin/bash

make clean
mkdir execs
rm execs/*
cache_l2_block_sizes_M="120"
cache_l2_block_sizes_N="120"
cache_l2_block_sizes_K="160 240"
cache_block_sizes_M="30 40"
cache_block_sizes_N="30 40"
cache_block_sizes_K="40 80 160"
reg_block_sizes_M="2"
reg_block_sizes_N="2"
reg_block_sizes_K="8"

for ii in $cache_l2_block_sizes_M
do
  for ii2 in $cache_l2_block_sizes_N
  do
    for ii3 in $cache_l2_block_sizes_K 
    do
for i in $cache_block_sizes_M
do
  for i2 in $cache_block_sizes_N 
  do
    for i3 in $cache_block_sizes_K 
    do
      for j in $reg_block_sizes_M
      do
	for k in $reg_block_sizes_N 
	do
	  for l in $reg_block_sizes_K 
	  do
	    make DFLAGS="-DBLOCK_SIZE_M=$i -DBLOCK_SIZE_N=$i2 -DBLOCK_SIZE_K=$i3 -DRBM=$j -DRBN=$k -DRBK=$l -DL2M=$ii -DL2N=$ii2 -DL2K=$ii3"
	    cp fmm_test execs/bench_$i\_$i2\_$i3\_$j\_$k\_$l\_$ii\_$ii2\_$ii3
	    make clean
	  done
	done
      done
    done
  done
done
done
done
done

