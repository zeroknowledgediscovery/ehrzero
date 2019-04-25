set grid
#set logscale y
plot "res1.txt" u 1:2 w lp ps 2 pt 7 lw 3 lt 2 smooth bezier  notitle
replot "res2.txt" u 1:2 w lp ps 2 pt 7 lw 3 lt 3 smooth bezier notitle
replot "res3.txt" u 1:2 w lp ps 2 pt 7 lw 3 lt 4 smooth bezier notitle
replot "res4.txt" u 1:2 w lp ps 2 pt 7 lw 3 lt 5 smooth bezier notitle
replot "ren1.txt" u 1:2 w lp ps 2 pt 7 lw 3 lt 6 smooth bezier notitle
replot "ren2.txt" u 1:2 w lp ps 2 pt 7 lw 3 lt 8 smooth bezier notitle
replot "ren3.txt" u 1:2 w lp ps 2 pt 7 lw 3 lt 9 smooth bezier notitle
replot "ren4.txt" u 1:2 w lp ps 2 pt 7 lw 3 lt 1 smooth bezier notitle

while (1) {

	replot
	pause 1
}
