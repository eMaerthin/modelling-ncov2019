push!(LOAD_PATH, "../julia/src/Simulation")
using JSON
using ArgParse
using Random
using FileIO
using JLD2
using DataFrames
import Simulation

function parse_commandline()
	s = ArgParseSettings()
	@add_arg_table! s begin
		"--JSON"
		help = "path to a JSON file with the parameter settings. If parameters are given sepeartely, the parameters of the JSON are overwritten in the program."
		"--output_file"
		help = "path to the output file"
	end
return parse_args(s)
end

function read_params(json, rng::AbstractRNG)
  constant_kernel_param = json["transmission_probabilities"]["constant"]  |> Float64
  household_kernel_param = json["transmission_probabilities"]["household"] |> Float64
  mild_detection_prob = json["detection_mild_proba"]  |> Float64

  tracking_prob = json["contact_tracking"]["probability"]  |> Float64
  tracking_delay = json["contact_tracking"]["delay"]  |> Float64
  population_path = json["population_path"] # <= JSON
	
  individuals_df = load(population_path)["individuals_df"]

  Simulation.load_params(
    rng,
    population = individuals_df, 
        
    mild_detection_prob = mild_detection_prob,
        
    constant_kernel_param = constant_kernel_param,
    household_kernel_param = household_kernel_param,
        
    backward_tracking_prob = tracking_prob,
    backward_detection_delay = tracking_delay/2,
        
    forward_tracking_prob = tracking_prob,
    forward_detection_delay = tracking_delay/2,
        
    testing_time = tracking_delay/2
)
end

function main()
  parsed_args = parse_commandline()
  # parse the JSON file if provided
  if parsed_args["JSON"] === nothing
    println("give me sim_params, option: --JSON path")
    exit()
  else
    param_dict = JSON.parsefile(parsed_args["JSON"])
    rng = MersenneTwister()
    max_num_infected = param_dict["stop_simulation_threshold"]
	
    num_trajectories = param_dict["random_seed"] 
    num_initial_infected = param_dict["initial_conditions"]["cardinalities"]["infectious"]
    parameters = read_params(param_dict, rng) 
  end
	
  if parsed_args["output_file"] === nothing
    output_path = "output.jld2"
  else
    output_path = parsed_args["output_file"]
  end

  state = Simulation.SimState(
    rng,
    length(parameters.household_ptrs) # number of individuals to allocate for
  ); 

  jldopen(output_path, "w") do file
    infection_times = Vector{Float32}()
    sizehint!(infection_times, max_num_infected)

    detection_times = Vector{Float32}()
    sizehint!(detection_times, max_num_infected)
		
    function callback(event::Simulation.Event, state::Simulation.SimState, params::Simulation.SimParams)
      eventkind = Simulation.kind(event)
      if eventkind == Simulation.DetectedOutsideQuarantineEvent || 
          eventkind == Simulation.DetectedFromTrackingEvent ||
          eventkind == Simulation.DetectedFromQuarantineEvent
        push!(detection_times, Simulation.time(event))
      elseif eventkind == Simulation.TransmissionEvent || 
          eventkind == Simulation.OutsideInfectionEvent
        push!(infection_times, Simulation.time(event))
      end

      return max_num_infected >= length(infection_times)
    end

    for trajectory_id in 1:num_trajectories
      empty!(infection_times)
      empty!(detection_times)

      seed = trajectory_id
      Simulation.reset!(state, seed)
      Simulation.initialfeed!(state, num_initial_infected)
      Simulation.simulate!(state, parameters, callback)

      trajectory_group = JLD2.Group(file, string(trajectory_id))
      trajectory_group["infection_times"] = infection_times 
      trajectory_group["detection_times"] = detection_times
    end
  end
end

main()
