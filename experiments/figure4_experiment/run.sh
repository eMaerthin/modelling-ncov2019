#!/bin/bash

PROJDIR=$PWD

individuals_path=$PROJDIR/data/raw/poland_population/aggregated_population.csv
households_path=$PROJDIR/data/raw/poland_population/aggregated_households.csv

for i in $PROJDIR/experiments/figure4_experiment/figure*;
do
  python $PROJDIR/src/models/infection_model.py --params-path $i/data/params_experiment.json --df-individuals-path $individuals_path --df-households-path $households_path run-simulation
done