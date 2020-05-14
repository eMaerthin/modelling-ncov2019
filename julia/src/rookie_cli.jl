push!(LOAD_PATH, "Simulation")
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
  constant_kernel_param = json["transmission_probabilities"]["constant"]  |> Float64
  household_kernel_param = json["transmission_probabilities"]["household"] |> Float64
  hospital_kernel_param = get(json["transmission_probabilities"],"hospital", 0.0) |> Float64
  
  mild_detection_prob = json["detection_mild_proba"]  |> Float64

  tracking_prob = json["contact_tracking"]["probability"]  |> Float64
  tracking_backward_delay = json["contact_tracking"]["backward_detection_delay"]  |> Float64
  tracking_forward_delay = json["contact_tracking"]["forward_detection_delay"]  |> Float64
  tracking_testing_delay = json["contact_tracking"]["testing_time"]  |> Float64

  phone_tracking_usage = json["phone_tracking"]["usage"] |> Float64
  phone_tracking_testing_delay = json["phone_tracking"]["detection_delay"] |> Float64

  population_path = json["population_path"] # <= JSON
  population_path::AbstractString # checks if it was indeed a string

  individuals_df = load(population_path)["individuals_df"]

  Simulation.load_params(
    rng,
    population = individuals_df, 
        
    mild_detection_prob = mild_detection_prob,
        
    constant_kernel_param = constant_kernel_param,
    household_kernel_param = household_kernel_param,
    hospital_kernel_param = hospital_kernel_param,
        
    backward_tracking_prob = tracking_prob,
    backward_detection_delay = tracking_backward_delay,
        
    forward_tracking_prob = tracking_prob,
    forward_detection_delay = tracking_forward_delay,
        
    testing_time = tracking_testing_delay,

    phone_tracking_usage = phone_tracking_usage,
    phone_detection_delay = phone_tracking_testing_delay
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

isdetection(ek::Simulation.EventKind) = ek == Simulation.DetectedOutsideQuarantineEvent || ek == Simulation.DetectedFromTrackingEvent || ek == Simulation.DetectedFromQuarantineEvent
istransmission(ek::Simulation.EventKind) = ek == Simulation.TransmissionEvent || ek == Simulation.OutsideInfectionEvent
isquarantine(ek::Simulation.EventKind) = ek == Simulation.QuarantinedEvent || ek == Simulation.QuarantineEndEvent
ishospitalized(ek::Simulation.EventKind) = ek == Simulation.GoHospitalEvent
isdeath(ek::Simulation.EventKind) = ek == Simulation.DeathEvent

function (cb::DetectionCallback)(event::Simulation.Event, state::Simulation.SimState, params::Simulation.SimParams)
  eventkind = Simulation.kind(event)
  contactkind = Simulation.contactkind(event)
  subject = Simulation.subject(event)
  if isdetection(eventkind)
    cb.detection_times[subject] = Simulation.time(event)
    if Simulation.DetectedOutsideQuarantineEvent == eventkind
      cb.detection_types[subject] = 1
    elseif Simulation.DetectedFromTrackingEvent == eventkind
      cb.detection_types[subject] = 2
    elseif Simulation.DetectedFromQuarantineEvent == eventkind
      cb.detection_types[subject] = 3
    end
  elseif istransmission(eventkind)
    cb.num_infected_remain -= 1
  elseif eventkind == Simulation.TrackedEvent
    cb.tracking_sources[subject] = Simulation.source(event)
  end
  return cb.num_infected_remain>0
end

const comma = ','    
const float_with_offset(offset, time) = float(offset + time)

function progression_csv_output(path::AbstractString, state::Simulation.SimState, params::Simulation.SimParams)
  io = open(path, "w")
  try    
    # print header
    println(io, "infection_time", comma, "subject_id", comma, "source_id", comma, "contact_kind", comma, "severity", comma, "incubation_time", comma, "mild_time", comma, "severe_time", comma, "recovery_time", comma, "death_time")
          
    for (e, p) in zip(state.forest.inedges, params.progressions)
      if Simulation.NoContact == e.contact_kind
        continue
      end
      t = Simulation.time(e)
      
      # print row
      println(io, float(t), comma, e.subject_id, comma, e.source_id, comma, UInt8(e.contact_kind),
                  comma, p.severity, comma, float(t+p.incubation_time), comma, float_with_offset(t, p.mild_symptoms_time),
                  comma, float_with_offset(t, p.severe_symptoms_time), comma, float_with_offset(t, p.recovery_time), 
                  comma, float_with_offset(t, p.death_time))
    end
  finally
    close(io)  
  end
  nothing
end

function callback_csv_output(path::AbstractString, cb::DetectionCallback)
  io = open(path, "w")
  try
    println(io, "subject_id", comma, "detection_time", comma, "detection_type", comma, "tracking_source_id")
    for (i, (time_detection, type_detection, tracking_source)) in enumerate(zip(cb.detection_times, cb.detection_types, cb.tracking_sources))
      if 0 == type_detection
        continue
      end
      println(io, i, comma, Float32(time_detection), comma, type_detection, comma, tracking_source)
    end
  finally
    close(io)
  end
  nothing
end

function main()
  parsed_args = parse_commandline()
  # parse the JSON file if provided
  if parsed_args["JSON"] === nothing
    println("give me sim_params, option: --JSON path")
    exit()
  end
    
  json = JSON.parsefile(parsed_args["JSON"])
  rng = MersenneTwister(11)
  max_num_infected = json["stop_simulation_threshold"] |> Int
	
  num_trajectories = json["num_trajectories"] |> Int
  num_initial_infected = json["initial_conditions"]["cardinalities"]["infectious"] |> Int
  params = read_params(json, rng) 
    
  output_prefix = ""
  if parsed_args["output_prefix"] !== nothing
    output_prefix = parsed_args["output_prefix"]
  end

  state = Simulation.SimState(
    rng,
    Simulation.num_individuals(params) # number of individuals to allocate for
  ); 

  @showprogress for trajectory_id in 1:num_trajectories
    Simulation.reset!(state)
    Simulation.initialfeed!(state, num_initial_infected)

    callback = DetectionCallback(Simulation.num_individuals(params), max_num_infected) #TODO reset!(cb)
    try
      Simulation.simulate!(state, params, callback)
    catch err
      println(stderr, "iteration ", trajectory_id, " failed: ", err)
      foreach( println, stacktrace(catch_backtrace()))
    end
    
    GC.gc()
    
    progression_csv_output(output_prefix*"progressions_$(trajectory_id).csv", state, params)
    callback_csv_output(output_prefix*"detections_$(trajectory_id).csv", callback)
  end
end

main()
