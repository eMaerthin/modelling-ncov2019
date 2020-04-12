module Simulation

__precompile__(true)

using CSV
using DataFrames

using Distributions
using GZip
using JSON
using LinearAlgebra
using NPZ
using ProgressMeter
using Random
using Setfield
using StaticArrays

using FixedPointNumbers

TimePoint = Fixed{Int32, 16}
TimeDiff = Fixed{Int32, 16}



include("enums.jl")
include("event.jl")

include("eventqueue.jl")
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
        t0_to_t2_samples_path::AbstractString,
        
        kwargs...
        )
  
  individuals_df = load_individuals(population_path)
  num_individuals = individuals_df |> nrow

  dist_severity = Categorical([0/10, 7/10, 2/10, 1/10])
  dist_incubation_time = load_dist_from_samples(incubation_time_samples_path)
  dist_symptom_onset_time = load_dist_from_samples(t0_to_t1_samples_path)
  dist_hospitalization_time = load_dist_from_samples(t0_to_t2_samples_path)

  progressions = 1:num_individuals .|> _ -> sample_progression(rng, 
    dist_severity, 
    dist_incubation_time, 
    dist_symptom_onset_time, 
    dist_hospitalization_time)
  
  make_params(rng, individuals_df=individuals_df, progressions=progressions, kwargs...)
end

function make_params(rng=MersenneTwister(0);
        individuals_df::DataFrame,
        progressions::AbstractArray{Progression},

        constant_kernel_param::Float64=1.0,
        household_kernel_param::Float64=1.0,
        
        backward_tracking_prob::Float64=1.0,
        backward_detection_delay::Float64=1.0,
        
        forward_tracking_prob::Float64=1.0,
        forward_detection_delay::Float64=1.0,
        
        quarantine_length::Float64=14.0,
        testing_time::Float64=1.0
        )
  sort!(individuals_df, :household_index)

  num_individuals = individuals_df |> nrow
    
  @assert num_individuals == length(progressions)

  household_ptrs = collect( zip(groupptrs(individuals_df.household_index)...))
  
  params = SimParams(
    household_ptrs,
    progressions,        
    constant_kernel_param,   
    household_kernel_param,
    
    backward_tracking_prob,
    backward_detection_delay,
    
    forward_tracking_prob,
    forward_detection_delay,
    
    quarantine_length, # quarantine length
    testing_time # testing time
  )
  params
end

function simulate!(state::SimState, 
                  params::SimParams; 
                  history::Union{Nothing, Vector{Event}}=nothing, 
                  execution_history::Union{Nothing, BitVector}=nothing,
                  state_history::Union{Nothing, Vector{IndividualState}}=nothing,
                  )
  iter_no = 0
  while true
    if isempty(state.queue)
      println("Empty queue after $iter_no events ")
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
    if nothing !== history
      push!(history, event)
    end
    @assert state.time <= time(event)  "time for event $event was smaller than current time $(state.time)"
    state.time = time(event)
      
    result::Bool = execute!(state, params, event)
      
    if nothing !== execution_history
      push!(execution_history, result)
    end  
    
    if nothing !== state_history
      push!(state_history, state.individuals[subject(event)])
    end
    
    iter_no+=1
  end
  nothing
end
  
  
end