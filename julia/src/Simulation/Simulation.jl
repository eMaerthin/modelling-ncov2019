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

const TimePoint = Fixed{Int32, 16}
const TimeDiff = Fixed{Int32, 16}

include("enums.jl")
include("event.jl")

include("eventqueue.jl")
include("infection_forest.jl")
include("simstate.jl")
include("progression.jl")
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
                   state_history::Union{Nothing, Vector{IndividualState}}=nothing)
                         
  iter_no = 0
  while true
    if isempty(state.queue)
      #println("Empty queue after $iter_no events ")
      break
    end
      
    event = pop!(state.queue)
        
    if nothing !== history
      push!(history, event)
    end
      
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
  
using FunctionWrappers
#const Callback = FunctionWrappers.FunctionWrapper{Nothing, Tuple{Event, SimState, SimParams}}

function simulate!(state::SimState, params::SimParams, callback::Union{Nothing,Function})
  iter_no = 0
  while true
    if isempty(state.queue)
      #println("Empty queue after $iter_no events ")
      break
    end
      
    event = pop!(state.queue)
      
    executed::Bool = execute!(state, params, event)
      
    if executed
       should_continue = callback(event, state, params)
    end  
    
    iter_no+=1
  end
  nothing
end

  
end #module