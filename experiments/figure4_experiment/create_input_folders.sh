#!/bin/bash
step=0.01
range=12
arr=("1000" "1196" "1430" "1711" "2046" "2447" "2927" "3501" "4187" "5008" "5990" "7164" "8569" "10248" "12257" "14660" "17534" "20971" "25082" "29999")
arr_size=20
for (( i=0; i<=$range; i++ ))
do
	for (( j=0; j<$arr_size; j++ ))
	do
		rm -f template/_SUCCESS
		cp -r template "figure4_"$i"_"$j
		a=`awk "BEGIN {print $step*$i}"`
		b=${arr[$j]}
		echo "params_experiment_"$i"_"$j".json"
		cat template/data/template.json | sed "s/constant\": 0.0/constant\": $a/" > "figure4_"$i"_"$j"/data/params_experiment_"$i"_"$j"_1.json"
		cat "figure4_"$i"_"$j"/data/params_experiment_"$i"_"$j"_1.json" | sed "s/contraction\": 1000/contraction\": $b/" > "figure4_"$i"_"$j"/data/params_experiment.json"
                rm "figure4_"$i"_"$j"/data/params_experiment_"$i"_"$j"_1.json"
                rm "figure4_"$i"_"$j"/data/template.json"
		touch template/_SUCCESS
	done
done
