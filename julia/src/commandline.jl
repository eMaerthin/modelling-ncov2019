using JSON
using ArgParse
using Random
using Distributions
using PyPlot
include("Simulation/Simulation.jl") # This is somehow compiled every time new. So takes time
# This is a commandline front end for the Simulations in Julia
# The behavior is as follows.
# First the JSON is loaded. Any addtional arguments passed will overwrite the JSON setting.
# Arguments that have no corresponding support in the code give back a warning.

# Todo
# Add custom argtypes: https://argparsejl.readthedocs.io/en/latest/argparse.html#argparse-custom-parsing
# hide not used arguments

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--JSON"
            help = "path to a JSON file with the parameter settings. If parameters are given sepeartely, the parameters of the JSON are overwritten in the program."
        "--individuals_df"
	    help = "GZIP file containing a csv with individual level data"
        "--hospital_detections"
            help = "Are the patients detected when in hospital, Boolean true/false"
        "--backward_tracking_prob"
            help = "backward tracking probbability in [0,1]."
        "--backward_detection_delay"
            help = "delay of backtracking in days"
        "--forward_tracking_prob"
            help = "forward tracking probability in [0,1]"
        "--forward_detection_delay"
            help = "delay in forward tracking in days."
        "--debug_return"
            help = "should debuggin information be printed? Boolean true/false"
            default = "false"
        "--turn_on_detection"
	    help = "should detection be turned on? boolean true/false"
            #arg_type = Int
	"--enable_visualization"
	    help = "should graphs be printed? boolean true/false"
	"--random_seed" 
            help="which random seeds should be used? One simulation per random seed is performed. E.g. 'range(10,20)'"
	"--output_root_dir"
	    help="path to the output directory"
	"--experiment_id"
	    help="name of the simulation"
	"--save_input_data"
	    help = "should the input data be saved? boolean true/false"
	"--max_time"
            help = "for how many days should the simulation run at most?"
            arg_type = Int
	"--log_outputs"
            help = "should output be logged? boolean true/false"
	"--log_time_freq"
	    help = "with which frequency logs should be printed? Integer"
	    arg_type = Int
	"--detection_mild_proba" 
            help = "probability of detecting an infection with mild progression"
	"--transmission_probabilities"
            help = "Parameters for the transmission probability. E.g. \"'friendship':0,'household':0.3,'constant':1.66\""
        "--fear_factors"
	   help = "Parameter for fear factors. E.g. \"'default':{'fear_function':'fear_disabled'}\""
        "--initial_conditions"
	   help = " E.g. \"'selection_algorithm':'random_selection', 'cardinalities':{'immune':0.32,'contraction':10}\""
	"--import_intensity"
	   help = "How are imports treated? E.g. \"'function':'no_import'\""
	"--case_severity_distribution"
           help = "Distribution of the differen severities. E.g. \"'asymptomatic':0.006,'mild':0.809,'severe':0.138,'critical':0.047\""
	"--death_probability"
	   help = "E.g. \"'asymptomatic':0,'mild':0,'severe':0,'critical':0.49\""
        "--epidemic_status"
           help = "'not_detected'"
	"--stop_simulation_threshold"
           help = "with how many infected the simulation should be stopped?"
           arg_type = Int
	"--disease_progression" 
           help = "E.g. \"'default':{ 't0':{'distribution':'from_file','filepath':'absolutepathto/incubation_period_distribution.npy','approximate_distribution':'lognormal'}, 't1':{'distribution':'from_file','filepath':'absolutepathto/t1_distribution.npy','approximate_distribution':'gamma'}, 't2':{'distribution':'from_file','filepath':'absolutepathto/t1_t2_distribution.npy','approximate_distribution':'gamma'}, 'tdeath':{'distribution':'from_file','filepath':'absolutepathto/onset_death_distribution.npy','approximate_distribution':'lognormal'} }\""
	"--icu_availability"
	   help= "number of available intensive care units beds with respiratory aid. "
	   arg_type= Int
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
#    println("Parsed args:")
#    for (arg,val) in parsed_args
#        println("  $arg  =>  $val")
#    end
    
    # parse the JSON file if provided
    if parsed_args["JSON"]===nothing
	parameters = Dict()
    else
	parameters = JSON.parsefile(parsed_args["JSON"])
    end
    
    # Overwrite Arguments from JSON by manually added Arguments
    parsed_args_b = filter( p -> !(last(p)===nothing), parsed_args)
    parameters = merge(parameters,parsed_args_b)

    # load individual data frame
    individuals_df = Simulation.load_individuals(parsed_args["individuals_df"]);
    
    # Do all random key independent stuff

    function simple_run(;tracking_prob::Float64, constant_kernel_param::Float64=1.0, 
            hospital_detections=true,
            history::Union{Nothing, Vector{Simulation.Event}}=nothing,
            execution_history::Union{Nothing, BitVector}=nothing,
            state_history::Union{Nothing, Vector{Simulation.IndividualState}}=nothing,
            debug_return=false,
            seed=123,
            initial_infections::Integer=1
        )
        rng = MersenneTwister(seed)
        params = Simulation.load_params(
            rng,
            population=individuals_df, 
            
            constant_kernel_param=constant_kernel_param,
            household_kernel_param=1.0,
            
            hospital_detections=hospital_detections,
            
            backward_tracking_prob=tracking_prob,
            backward_detection_delay=0.25,
            
            forward_tracking_prob=tracking_prob,
            forward_detection_delay=0.25,
            
            testing_time=0.25
        );
        state = Simulation.SimState(params.progressions |> length, seed=seed)
        
        sample(rng, 1:length(params.progressions), initial_infections) .|> person_id -> 
            push!(state.queue, Simulation.Event(Val(Simulation.OutsideInfectionEvent), 0.0, person_id))
    
        @time Simulation.simulate!(
            state, 
            params, 
            history=history, 
            execution_history=execution_history, 
            state_history=state_history
        )
        return state
    end
    
    function merge_history(history::Vector{Simulation.Event}, execution_history::BitVector, state_history::Vector{Simulation.IndividualState}; pop_last_event=false)
        history = copy(history)
        if pop_last_event
            last_event = pop!(history)
        end
        zip(history, state_history, execution_history)
    end
    history = Vector{Simulation.Event}()
    state_history = Vector{Simulation.IndividualState}()
    execution_history=BitVector()
    
    state = simple_run(tracking_prob=0.0,
        hospital_detections=false,
        history=history,
        execution_history=execution_history,
        state_history=state_history,
        initial_infections=10
    );
    function state2plot(state::Simulation.SimState)
        transmissions = vcat(state.forest.infections...)
        sort!(transmissions, lt=(x,y)->x.time<y.time)
        times = getproperty.(transmissions, :time)
        points = 1:length(transmissions)
        return times, points
    end
    times, points = state2plot(state)  
    plot(times, points)
    xlabel("time")
    ylabel("infections")

end

main()

