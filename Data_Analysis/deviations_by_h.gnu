#!/usr/bin/gnuplot -c


#This script takes in three input paramters
#It can be invoked with the command
#gnuplot -c script_name filein fileout lineno
#The input arguments have the following meaning
#filein: name of the file with input data
#fileout: name of the file to which the plot is meant to be written
#titlename: The title of the plot to be output
#lineno: Number of lines in the infile


filein=ARG1
fileout=ARG2
title_name=ARG3
lineno=ARG4

set terminal pdf
set output fileout
set xrange [1:lineno]
set grid
set title title_name
set xlabel '-log_2(h)'
set ylabel "Deviation from true solution"
set y2label "Energy norm"
#unset kee
var = lineno/9
set xtics var
set ytics nomirror
set format y "%.1e"
set y2tics
set format y2 "%1.e"
set logscale y
set logscale y2

f(x) = 0.64118*2**(-2*x)
en(x) = 4.96656*2**(-x)


#plot filein using 1:2 axes x1y1 with lines title 'deviations', \ 
#filein using 1: 0.6*2**(-2*$1) axes x1y1 with lines title 'ch^2'

plot filein using 1:2 axes x1y1 with points ls 2 ps 0.75 lc rgb "blue" title 'deviations', \
filein using 1:(f($1)) axes x1y1 with lines title '0.64118h^2', \
filein using 1:3 axes x1y2 with points ls 7 ps .4 title 'energy norm', \
filein using 1:(en($1)) axes x1y2 with lines title '2.48328h'

