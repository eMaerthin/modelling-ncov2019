#!/bin/bash

PROJDIR=$PWD

FIGURE_CSV=$PROJDIR/experiments/figure4_experiment/figure4.csv
touch $FIGURE_CSV
echo "Mean_Time;Mean_Affected;Wins_freq;c;c_norm;Init_people" > $FIGURE_CSV
for i in $PROJDIR/experiments/figure4_experiment/figure*/outputs/under_critical/aggregated*/results_mean.txt;
do
  tail -n +2 $i >> $FIGURE_CSV
done