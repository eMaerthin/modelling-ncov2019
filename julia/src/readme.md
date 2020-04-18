
# This Commandline tool allows the usage of the JSON files used for the Python code.

With

```
julia commandline.jl --help
```

All possible arguments are printed with description.

For example one can call this with a specific JSON and an specific individuals\_df

```
julia commandline.jl --JSON /path/to/JSON --individuals_df ../../data/simulations/mini-population.csv.gz
```

The arguments defined in the JSON are automatically overwritten by those given additionally. So in the example above, the individuals\_df argument in the JSON would be changed to ../../data/simulations/mini-population.csv.gz .
This is the first version and actually does not pass the arguments to the simulation yet. This has to be improved still.
