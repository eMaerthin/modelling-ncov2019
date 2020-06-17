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
  hospital_kernel_param = get(json["transmission_probabilities"], "hospital", 0.0) |> float
  friendship_kernel_param = get(json["transmission_probabilities"], "friendship", 0.0) |> float

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

  spreading = get(json, "spreading", nothing)
  spreading_alpha = isnothing(spreading) ? nothing : spreading["alpha"]
  spreading_x0 = isnothing(spreading) ? 1 : get(spreading, "x0", 1)
  spreading_truncation = isnothing(spreading) ? Inf : get(spreading, "truncation", Inf)

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
    infection_modulation_params=infection_modulation_params,

    spreading_alpha=spreading_alpha,
    spreading_x0=spreading_x0,
    spreading_truncation=spreading_truncation
)
end
const OptTimePoint = Union{Missing, Simulation.TimePoint}

struct DetectionCallback
    detection_times::Vector{OptTimePoint}
    detection_types::Vector{UInt8}

    tracking_times::Vector{OptTimePoint}
    tracking_sources::Vector{UInt32}
    tracking_types::Vector{UInt8}
    
    max_num_infected::UInt32
end
DetectionCallback(sz::Integer, max_num_infected::Integer=10^8) = DetectionCallback(
    Vector{OptTimePoint}(missing, sz),
    fill(UInt8(0), sz),
    Vector{OptTimePoint}(missing, sz),
    fill(UInt32(0), sz),
    fill(UInt8(0), sz),
    max_num_infected
)

optreal2float(optreal::Union{Missing,T} where T<:Real) = ismissing(optreal) ? NaN32 : Float32(optreal)

function saveparams(dict, cb::DetectionCallback, prefix::AbstractString="") 
  dict[prefix*"detection_times"] = optreal2float.(cb.detection_times)
  dict[prefix*"detection_types"] = cb.detection_types

  dict[prefix*"tracking_times"] = optreal2float.(cb.tracking_times)
  dict[prefix*"tracking_sources"] = cb.tracking_sources
  dict[prefix*"tracking_types"] = cb.tracking_types
end

function reset!(cb::DetectionCallback)
  fill!(cb.detection_times, missing)
  fill!(cb.detection_types, 0)
  fill!(cb.tracking_sources, 0)
  fill!(cb.tracking_types, 0)
end

function (cb::DetectionCallback)(event::Simulation.Event, state::Simulation.SimState, params::Simulation.SimParams)
  eventkind = Simulation.kind(event)
  contactkind = Simulation.contactkind(event)
  subject = Simulation.subject(event)
  if Simulation.isdetection(eventkind)
    cb.detection_times[subject] = Simulation.time(event)
    cb.detection_types[subject] = Simulation.detectionkind(event) |> UInt8
  elseif Simulation.istracking(eventkind)
    cb.tracking_times[subject] = Simulation.time(event)
    cb.tracking_sources[subject] = Simulation.source(event)
    cb.tracking_types[subject] = Simulation.trackingkind(event) |> UInt8
  end
  return Simulation.numinfected(state.stats) < cb.max_num_infected
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
  #@threads 
  for trajectory_id in 1:num_trajectories
    state = states[threadid()]
    Simulation.reset!(state, trajectory_id)
    Simulation.initialfeed!(state, num_initial_infected)

    callback = callbacks[threadid()]
    reset!(callback)
    #try
      Simulation.simulate!(state, params, callback)
      try
        lock(writelock) # JLD2 is not thread-safe, not even when files are separate
        save_infections_and_detections(output_prefix*"run_$trajectory_id.jld2", state, callback)
      finally
        GC.gc()
        unlock(writelock)
      end
    #catch err
    #  println(stderr, "Failed on thread ", threadid(), " iteration ", trajectory_id, " failed: ", err)
    #  foreach(x -> println(stderr, x), stacktrace(catch_backtrace()))
    #end
        
    ProgressMeter.next!(progress) # is thread-safe
  end
end

main()
