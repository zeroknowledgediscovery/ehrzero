#!/bin/bash

FILES='exD1.dx  exD2.dx  exD3.dx  exD4.dx  exN1.dx  exN2.dx  exN3.dx  exN4.dx'
rm -rf *txt

#for i in {10..150}
#do
#    for fl in $FILES
#    do
#	./zero.py -d $fl -n $i >> "$fl"'.txt'
#    done
#done
cd ..
PROG="./zero.py "
parallel -j 10 $PROG -d ./tests/{1} -n {2} -B ./tests/{1}.txt ::: $FILES ::: `seq 10 150`
