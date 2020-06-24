module Launcher

using ArgParse
using Base.Threads
using DataFrames
using FileIO
using JLD2
using JSON
using ProgressMeter
using Random

import Simulation
import Simulation: ContactKind, NoContact, contactkind, time

const OptTimePoint = Union{Missing, Simulation.TimePoint}
optreal2float32(optreal::Union{Missing,T} where T<:Real) = ismissing(optreal) ? NaN32 : Float32(optreal)

include("cmd_parsing.jl")
include("callback.jl")
include("load_params.jl")
include("outputs.jl")

export launch

function launch()
  @info "Stated" nthreads()
  if nthreads() == 1
    @warn "using single thread, set more threads by setting JULIA_NUM_THREADS environment variable"
  end

  cmd_args = parse_commandline()
  @info "Parsed args" cmd_args
  json = JSON.parsefile(cmd_args["JSON"])

  max_num_infected = json["stop_simulation_threshold"] |> Int
  num_trajectories = json["num_trajectories"] |> Int
  num_initial_infected = json["initial_conditions"]["cardinalities"]["infectious"] |> Int
  params_seed = get(json, "params_seed", 0)

  @info "loading population and setting up parameters" params_seed
  rng = MersenneTwister(params_seed)
  params = read_params(json, rng) 
  num_individuals =  Simulation.numindividuals(params)
  
  states = [Simulation.SimState(num_individuals) for _ in 1:nthreads()]
  callbacks = [DetectionCallback(num_individuals, max_num_infected) for _ in 1:nthreads()]

  outputs = make_outputs(cmd_args, num_trajectories)
  foreach(o->beforetrajectories(o, params), outputs)

  @info "starting simulation" num_trajectories
  writelock = ReentrantLock()
  progress = ProgressMeter.Progress(num_trajectories)
  GC.gc()
  @threads for trajectory_id in 1:num_trajectories
    state = states[threadid()]
    Simulation.reset!(state, trajectory_id)
    Simulation.initialfeed!(state, num_initial_infected)

    callback = callbacks[threadid()]
    reset!(callback)
    try
      Simulation.simulate!(state, params, callback)
      try
        lock(writelock) # JLD2 is not thread-safe, not even when files are separate
        foreach(o->pushtrajectory!(o, state, params, callback), outputs)
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
  foreach(o->aftertrajectories(o, params), outputs)
end

end