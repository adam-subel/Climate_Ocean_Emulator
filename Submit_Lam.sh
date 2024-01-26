#!/bin/bash

for i in Quiescent_Ext
do
    for j in 12  
    do
        for k in 5 
        do
	    for m in 1
            do
                sbatch batch_test.batch 3 $j 2 region $i rand_seed $m lag $k
                echo $i $j $k $m
	    done
        done
    done
done
