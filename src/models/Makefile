delli=/home/matteo/Projects/corona/modelling-ncov2019
czczmiel=/home/cov/git/modelling-ncov2019

run_delli:
	python model.py $(delli)/test/models/assets/params_experiment0.json $(delli)/data/vroclav/population_experiment0.csv $(delli)/data/vroclav/households_experiment0.csv

run_delli_original:
	python infection_model.py --params-path $(delli)/test/models/assets/params_experiment0.json --df-individuals-path $(delli)/data/vroclav/population_experiment0.csv --df-households-path $(delli)/data/vroclav/households_experiment0.csv run-simulation

profile_delli_original:
	python -m cProfile -o profile_experiment0.txt infection_model.py --params-path $(delli)/test/models/assets/params_experiment0.json --df-individuals-path $(delli)/data/vroclav/population_experiment0.csv --df-households-path $(delli)/data/vroclav/households_experiment0.csv run-simulation

run czczmiel:
	python model.py $(czczmiel)/test/models/assets/params_experiment0.json $(czczmiel)/data/vroclav/population_experiment0.csv $(czczmiel)/data/vroclav/households_experiment0.csv

run_czczmiel_original:
	python infection_model.py --params-path $(czczmiel)/test/models/assets/params_experiment0.json --df-individuals-path $(czczmiel)/data/vroclav/population_experiment0.csv --df-households-path $(czczmiel)/data/vroclav/households_experiment0.csv run-simulation

profile_czczmiel:
	python -m cProfile -o profile_experiment0.txt infection_model.py --params-path $(czczmiel)/test/models/assets/params_experiment0.json --df-individuals-path $(czczmiel)/data/vroclav/population_experiment0.csv --df-households-path $(czczmiel)/data/vroclav/households_experiment0.csv run-simulation


