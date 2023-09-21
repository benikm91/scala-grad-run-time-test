#!/bin/bash

for exp_id in 0 1 2 3 4 5 6 7 8 9  # experiment most outer loop to temporally most distribute
do
  for nRows in 8 11 16  # 8*8=64; 11*11=121 16*16=256
  do
    for n in 16 32 64 128 256 512 1024 2056 4096 8192 16384 32768 65536 131072
    do
        # sbt "runMain scalagrad.scalaGradPerformanceTestMain $exp_id $n $nRows"
        sbt "runMain scalagrad.scalaGradComplexPerformanceTestMain2 $exp_id $n $nRows"
    done
  done
done