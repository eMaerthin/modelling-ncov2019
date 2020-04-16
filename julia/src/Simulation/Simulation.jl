module Simulation

__precompile__(true)

using CSV
using DataFrames
using Distributions
using FixedPointNumbers
using LinearAlgebra
using ProgressMeter
using Random
using Setfield
using StaticArrays

TimePoint = Fixed{Int32, 16}
TimeDiff = Fixed{Int32, 16}

include("enums.jl")
include("event.jl")

include("eventqueue.jl")
include("infection_forest.jl")
include("simstate.jl")
include("simparams.jl")

include("event_execution.jl")
include("infection_kernels.jl")

include("utils.jl")
include("data_loading.jl")

#export load_params
#export make_params
#export simulate!



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