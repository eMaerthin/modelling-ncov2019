delli=/home/matteo/Projects/corona/modelling-ncov2019
czczmiel=/home/cov/git/modelling-ncov2019

run_simulation:
	python infection_model.py --params-path $(czczmiel)/test/models/assets/params_experiment0.json --df-individuals-path $(czczmiel)/data/vroclav/population_experiment0.csv --df-households-path $(czczmiel)/data/vroclav/households_experiment0.csv run-simulation

profile:
	python -m cProfile -o profile_experiment0.txt infection_model.py --params-path $(czczmiel)/test/models/assets/params_experiment0.json --df-individuals-path $(czczmiel)/data/vroclav/population_experiment0.csv --df-households-path $(czczmiel)/data/vroclav/households_experiment0.csv run-simulation

run_delli_original:
	python infection_model.py --params-path $(delli)/test/models/assets/params_experiment0.json --df-individuals-path $(delli)/data/vroclav/population_experiment0.csv --df-households-path $(delli)/data/vroclav/households_experiment0.csv run-simulation

run_delli:
	python model.py --params-path $(delli)/test/models/assets/params_experiment0.json --df-individuals-path $(delli)/data/vroclav/population_experiment0.csv --df-households-path $(delli)/data/vroclav/households_experiment0.csv run-simulation
