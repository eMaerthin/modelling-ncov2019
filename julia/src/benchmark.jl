#push!(LOAD_PATH,"Simulation")
#import Simulation
include("Simulation/Simulation.jl")

using Random
using Profile
using FileIO
using JLD2

using Distributions
using DataFrames


let 
  individuals_df = load("../../data/simulations/wroclaw-population-orig.jld2")["individuals_df"]

  if(!(:social_competence in names(individuals_df)))
    #Produce some test data for friendship kernel
    individuals_df.social_competence = [Float32(x) for x in rand(Beta(3.0, 3.0), nrow(individuals_df))]
  end

  #loads and pre-processes data that is constant during the simulation time

  mild_detection_prob = 0.1
  constant_kernel_param = 1.0
  household_kernel_param = 0.3
  tracking_prob = 0.3
  tracking_delay = 5

  param_rng = MersenneTwister(1)  
  
  params = Simulation.load_params(
    param_rng,
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
    
  state = Simulation.SimState(Simulation.num_individuals(params), seed=123)
  sizehint!(state.queue, 10^6)

  push!(state.queue, Simulation.Event(Val(Simulation.OutsideInfectionEvent),0.0,1))

  function run_some(num_iters::Integer)
    for iter in 1:num_iters
        if isempty(state.queue)
            @info "queue empty before real testing starts, after $iter events"
            break
        end
        event = pop!(state.queue)
        result = Simulation.execute!(state, params, event)
    end
  end

  #@code_warntype run_some(10000)

  run_some(10000)

  Profile.clear_malloc_data()

  run_some(10000)
end

