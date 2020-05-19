#!/bin/bash

PROJDIR=$PWD

individuals_path=$PROJDIR/data/raw/wroclaw_population/population_experiment.csv
households_path=$PROJDIR/data/raw/wroclaw_population/households_experiment.csv

for i in $PROJDIR/experiments/figure5_experiment/figure*;
do
  python3 $PROJDIR/src/models/infection_model.py --params-path $i/data/params_experiment.json --df-individuals-path $individuals_path --df-households-path $households_path run-simulation
done