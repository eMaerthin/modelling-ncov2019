#!/bin/bash
step_det=0.1
step_red=0.05
range_det=10
range_red=20
for (( i=0; i<=$range_det; i++ ))
do
        for (( j=0; j<=$range_red; j++ ))
        do
                rm -f template/_SUCCESS
                cp -r template "figure5_"$i"_"$j
                a=`awk "BEGIN {print $step_det*$i}"`
                b=`awk "BEGIN {print $step_red*$j}"`
                echo "params_experiment_"$i"_"$j".json"
                cat template/data/template.json | sed "s/detection_mild_proba\": 0.0/detection_mild_proba\": $a/" > "figure5_"$i"_"$j"/data/params_experiment_"$i"_"$j".json"
                cat "figure5_"$i"_"$j"/data/params_experiment_"$i"_"$j".json" | sed "s/limit_value\": 0.0/limit_value\": $b/" > "figure5_"$i"_"$j"/data/params_experiment.json"
                rm figure5_"$i"_"$j"/data/params_experiment_"$i"_"$j".json
                touch template/_SUCCESS
        done
done