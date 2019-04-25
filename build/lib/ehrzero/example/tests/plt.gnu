set grid
#set logscale y
plot "exD1.dx.txt"  u 1:2 w p ps 1.5 pt 7 lt 2 lw 2 notitle smooth bezier
replot "exD2.dx.txt"  u 1:2 w p ps 1.5 pt 7 lt 3 lw 2 notitle smooth bezier
replot "exD3.dx.txt"  u 1:2 w p ps 1.5 pt 7 lt 4 lw 2 notitle smooth bezier
replot "exD4.dx.txt"  u 1:2 w p ps 1.5 pt 7 lt 5 lw 2 notitle smooth bezier
replot "exN1.dx.txt"  u 1:2 w p ps 1.5 pt 7 lt 6 lw 2 notitle smooth bezier
replot "exN2.dx.txt"  u 1:2 w p ps 1.5 pt 7 lt 7 lw 2 notitle smooth bezier
replot "exN3.dx.txt"  u 1:2 w p ps 1.5 pt 7 lt 8 lw 2 notitle smooth bezier
replot "exN4.dx.txt"  u 1:2 w p ps 1.5 pt 7 lt 9 lw 2 notitle smooth bezier
pause 3
reread
