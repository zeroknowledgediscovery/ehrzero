#!/bin/bash

rm *.txt
cd ..


./zero.py -d tests/exD1.dx -n `seq 20 1 150` -B "tests/res1.txt"
./zero.py -d tests/exD2.dx -n `seq 20 1 150` -B "tests/res2.txt"
./zero.py -d tests/exD3.dx -n `seq 20 1 150` -B "tests/res3.txt"
./zero.py -d tests/exD4.dx -n `seq 20 1 150` -B "tests/res4.txt"
./zero.py -d tests/exN1.dx -n `seq 20 1 150` -B "tests/ren1.txt"
./zero.py -d tests/exN2.dx -n `seq 20 1 150` -B "tests/ren2.txt"
./zero.py -d tests/exN3.dx -n `seq 20 1 150` -B "tests/ren3.txt"
./zero.py -d tests/exN4.dx -n `seq 20 1 150` -B "tests/ren4.txt"

cd tests/

sed -i 's/ /\n/3; P; D' res1.txt
sed -i 's/ /\n/3; P; D' res2.txt
sed -i 's/ /\n/3; P; D' res3.txt
sed -i 's/ /\n/3; P; D' res4.txt
sed -i 's/ /\n/3; P; D' ren1.txt
sed -i 's/ /\n/3; P; D' ren2.txt
sed -i 's/ /\n/3; P; D' ren3.txt
sed -i 's/ /\n/3; P; D' ren4.txt

gnuplot gplot.gnu
