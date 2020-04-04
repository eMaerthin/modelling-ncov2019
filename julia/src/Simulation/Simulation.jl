module Simulation
  
using CSV
using DataFrames
using DataStructures
using Distributions
using GZip
using JSON
using LinearAlgebra
using Logging
using NPZ
using ProgressMeter
using PyPlot
using PyCall
using Random

import DataStructures.compare

include("enums.jl")
include("events.jl")

struct Earlier end
compare(c::Earlier, x::AbstractEvent, y::AbstractEvent) = time(x) < time(y)

include("simstate.jl")
include("simparams.jl")

include("event_execution.jl")
include("infection_kernels.jl")

include("utils.jl")
include("data_loading.jl")

export load_params
export simulate!

function load_params(rng=MersenneTwister(0);
        population_path::AbstractString,
        incubation_time_samples_path::AbstractString,
        t0_to_t1_samples_path::AbstractString,
        t0_to_t2_samples_path::AbstractString)
  
  individuals_df = load_individuals(population_path)
  num_individuals = individuals_df |> nrow

  dist_severity = Categorical([0/10, 7/10, 2/10, 1/10])
  dist_incubation_time = load_dist_from_samples(incubation_time_samples_path)
  dist_symptom_onset_time = load_dist_from_samples(t0_to_t1_samples_path)
  dist_hospitalization_time = load_dist_from_samples(t0_to_t2_samples_path)

  progression = 1:num_individuals .|> _ -> sample_progression(rng, 
    dist_severity, 
    dist_incubation_time, 
    dist_symptom_onset_time, 
    dist_hospitalization_time)
  
  household_ptrs = collect( zip(Simulation.groupptrs(individuals_df.household_index)...))
  #households = 1:num_individuals |> collect #TODO add household structure
  
  params = SimParams(
    household_ptrs,
    progression,        
    1.0,   
    
    1.0,
    2.0,
    
    1.0,
    2.0,
    
    14.0, # quarantine length
    2.0 # testing time
  )
  params
end

function simulate!(state::SimState, params::SimParams)
  iter_no = 0
  while true
    if isempty(state.queue)
      @info "Empty queue after $iter_no events "
      break
    end
      
    event = pop!(state.queue)
        #if state.affected_people >= params.stop_simulation_threshold
        #    @info "The outbreak reached a high number $(params.stop_simulation_threshold)"
        #    break
        #else
        #    event.time >= params.max_time
        #    @info "Max time reached"
        #    break
        #end
    @assert state.time <= time(event)  
    state.time = time(event)
      
    execute!(state, params, event)
      
    iter_no+=1
  end
end
  
  
end