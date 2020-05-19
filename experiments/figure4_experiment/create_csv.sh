#!/bin/bash

PROJDIR=$PWD

FIGURE_CSV=$PROJDIR/experiments/figure4_experiment/figure4.csv
touch $FIGURE_CSV
echo "Last_processed_time;Total_#Affected;Total_#Detected;Total_#Deceased;Total_#Quarantined;c;c_norm;Init_#people;Band_hit_time;Subcritical;runs;fear;detection_rate;incidents_per_last_day;over_icu;hospitalized;zero_time_offset;total_#immune" > $FIGURE_CSV
for i in $PROJDIR/experiments/figure4_experiment/figure*/outputs/under_critical/aggregated*/results.txt;
do
  tail -n +2 $i >> $FIGURE_CSV
done