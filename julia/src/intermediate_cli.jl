#using Pkg
#Pkg.activate(".")

push!(LOAD_PATH, "Simulation")
using Base.Threads
using JSON
using ArgParse
using Random
using FileIO
using JLD2
using DataFrames
using ProgressMeter

import Simulation

function parse_commandline()
	s = ArgParseSettings()
	@add_arg_table! s begin
		"--JSON"
		help = "path to a JSON file with the parameter settings. If parameters are given sepeartely, the parameters of the JSON are overwritten in the program."
		"--output_prefix"
		help = "path to the output file"
	end
return parse_args(s)
end

function read_params(json, rng::AbstractRNG)
  constant_kernel_param = json["transmission_probabilities"]["constant"]  |> float
  household_kernel_param = json["transmission_probabilities"]["household"] |> float
  hospital_kernel_param = get(json["transmission_probabilities"],"hospital", 0.0) |> float
  friendship_kernel_param = get(json["transmission_probabilities"],"friendship", 0.0) |> float

  mild_detection_prob = json["detection_mild_proba"]  |> float

  tracking_prob = json["contact_tracking"]["probability"]  |> float
  tracking_backward_delay = json["contact_tracking"]["backward_detection_delay"]  |> float
  tracking_forward_delay = json["contact_tracking"]["forward_detection_delay"]  |> float
  testing_time = json["contact_tracking"]["testing_time"]  |> float

  phone_tracking_usage = json["phone_tracking"]["usage"] |> float
  phone_tracking_testing_delay = json["phone_tracking"]["detection_delay"] |> float

  population_path = json["population_path"] # <= JSON
  population_path::AbstractString # checks if it was indeed a string

  individuals_df = load(population_path)["individuals_df"]

  infection_modulation_name, infection_modulation_params = if !haskey(json, "modulation")
    nothing, NamedTuple{}()
  else
    modulation = json["modulation"]
    params = get(modulation, "params", Dict{String,Any}())
    modulation["function"], NamedTuple{Tuple(Symbol.(keys(params)))}(values(params))
  end

  Simulation.load_params(
    rng,
    population = individuals_df, 
        
    mild_detection_prob = mild_detection_prob,
        
    constant_kernel_param = constant_kernel_param,
    household_kernel_param = household_kernel_param,
    hospital_kernel_param = hospital_kernel_param,
    friendship_kernel_param = friendship_kernel_param,
        
    backward_tracking_prob = tracking_prob,
    backward_detection_delay = tracking_backward_delay,
        
    forward_tracking_prob = tracking_prob,
    forward_detection_delay = tracking_forward_delay,
        
    testing_time = testing_time,

    phone_tracking_usage = phone_tracking_usage,
    phone_detection_delay = phone_tracking_testing_delay,

    infection_modulation_name=infection_modulation_name,
    infection_modulation_params=infection_modulation_params
)
end
const OptTimePoint = Union{Missing, Simulation.TimePoint}

mutable struct DetectionCallback
    detection_times::Vector{OptTimePoint}
    detection_types::Vector{UInt8}
    tracking_sources::Vector{UInt32}
    
    num_infected_remain::UInt32
end
DetectionCallback(sz::Integer, max_num_infected::Integer=10^8) = DetectionCallback(
    Vector{OptTimePoint}(missing, sz),
    fill(UInt8(0), sz),
    fill(UInt32(0), sz),
    max_num_infected
)
function saveparams(dict, cb::DetectionCallback, prefix::AbstractString="")
  N = length(cb.detection_times)
  detection_times = Vector{Float32}(undef, N)
  for i = 1:N
    timeopt = cb.detection_times[i]
    detection_times[i] = ismissing(timeopt) ? NaN32 : Float32(timeopt)
  end  
  dict[prefix*"detection_times"] = detection_times
  dict[prefix*"detection_types"] = cb.detection_types
  dict[prefix*"tracking_sources"] = cb.tracking_sources
end

function reset!(cb::DetectionCallback, max_num_infected::Integer=10^8)
  fill!(cb.detection_times, missing)
  fill!(cb.detection_types, 0)
  fill!(cb.tracking_sources, 0)
  cb.num_infected_remain = max_num_infected
end

function (cb::DetectionCallback)(event::Simulation.Event, state::Simulation.SimState, params::Simulation.SimParams)
  eventkind = Simulation.kind(event)
  contactkind = Simulation.contactkind(event)
  subject = Simulation.subject(event)
  if Simulation.isdetection(eventkind)
    cb.detection_times[subject] = Simulation.time(event)
    cb.detection_types[subject] = Simulation.detectionkind(event) |> UInt8
  elseif Simulation.istransmission(eventkind)
    cb.num_infected_remain -= 1
  elseif eventkind == Simulation.TrackedEvent
    cb.tracking_sources[subject] = Simulation.source(event)
  end
  return cb.num_infected_remain>0
end

function save_infections_and_detections(path::AbstractString, simstate::Simulation.SimState, callback::DetectionCallback)
  f = jldopen(path, "w", compress=true)
  try
    Simulation.saveparams(f, simstate)
    saveparams(f, callback)
  finally
    close(f)
  end
  nothing
end

function main()
  # set more threads by setting JULIA_NUM_THREADS environment variable
  @info "Stated" nthreads()

  parsed_args = parse_commandline()
  # parse the JSON file if provided
  if parsed_args["JSON"] === nothing
    println("give me sim_params, option: --JSON path")
    exit()
  end
    
  json = JSON.parsefile(parsed_args["JSON"])
  output_prefix = ""
  if parsed_args["output_prefix"] !== nothing
    output_prefix = parsed_args["output_prefix"]
  end

  max_num_infected = json["stop_simulation_threshold"] |> Int
  num_trajectories = json["num_trajectories"] |> Int
  num_initial_infected = json["initial_conditions"]["cardinalities"]["infectious"] |> Int
  params_seed = get(json, "params_seed", 0)

  @info "loading population and setting up parameters" params_seed
  rng = MersenneTwister(params_seed)
  params = read_params(json, rng) 
  num_individuals =  Simulation.numindividuals(params)
  param_save_path = output_prefix*"run_params.jld2"
  
  @info "saving parameters" num_individuals param_save_path
  jldopen(param_save_path, "w", compress=false) do dict
    Simulation.saveparams(dict, params)
  end

  states = [Simulation.SimState(num_individuals, seed=123) for _ in 1:nthreads()]
  callbacks = [DetectionCallback(num_individuals, max_num_infected) for _ in 1:nthreads()]

  @info "starting simulation" num_trajectories
  writelock = ReentrantLock()
  progress = ProgressMeter.Progress(num_trajectories)
  GC.gc()
  @threads for trajectory_id in 1:num_trajectories
    state = states[threadid()]
    Simulation.reset!(state, trajectory_id)
    Simulation.initialfeed!(state, num_initial_infected)

    callback = callbacks[threadid()]
    reset!(callback, max_num_infected)
    try
      Simulation.simulate!(state, params, callback)
      try
        lock(writelock) # JLD2 is not thread-safe, not even when files are separate
        save_infections_and_detections(output_prefix*"run_$trajectory_id.jld2", state, callback)
      finally
        GC.gc()
        unlock(writelock)
      end
    catch err
      println(stderr, "Failed on thread ", threadid(), " iteration ", trajectory_id, " failed: ", err)
      foreach(x -> println(stderr, x), stacktrace(catch_backtrace()))
    end
        
    ProgressMeter.next!(progress) # is thread-safe
  end
end

main()
