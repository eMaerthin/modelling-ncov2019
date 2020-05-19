#!/bin/bash

PROJDIR=$PWD

FIGURE_CSV=$PROJDIR/experiments/figure5_experiment/figure5.csv
touch $FIGURE_CSV
echo "Last_processed_time;Total_#Affected;Total_#Detected;Total_#Deceased;Total_#Quarantined;c;c_norm;Init_#people;Prevalence_30days;Prevalence_60days;Prevalence_90days;Prevalence_120days;Prevalence_150days;Prevalence_180days;Band_hit_time;Subcritical;Prevalence_360days;runs;fear;detection_rate;increase_10;increase_20;increase_30;increase_40;increase_50;increase_100;increase_150;incidents_per_last_day;over_icu;hospitalized" > $FIGURE_CSV
for i in $PROJDIR/experiments/figure5_experiment/figure*/outputs/figure5/aggregated*/results.txt;
do
  tail -n +2 $i >> $FIGURE_CSV
done