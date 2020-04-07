push!(LOAD_PATH,"Simulation")
import Simulation
#include("Simulation/Simulation.jl")

using Random
using Profile
#loads and pre-processes data that is constant during the simulation time
params = Simulation.load_params(
    MersenneTwister(3),
    population_path="../../data/simulations/wroclaw-population-orig.csv.gz", 
    incubation_time_samples_path="../../test/models/assets/incubation_period_distribution.npy", 
    t0_to_t1_samples_path="../../test/models/assets/t1_distribution.npy",
    t0_to_t2_samples_path="../../test/models/assets/t1_t2_distribution.npy");
    
state = Simulation.SimState(params.progressions |> length, seed=123)

push!(state.queue, Simulation.Event(Val(Simulation.OutsideInfectionEvent),0.0,1))


for iter  in 0:100
    #println("iteration=$iter")
    if isempty(state.queue)
        @info "Empty queue after $iter events"
        break
    end
    #state.individuals[1] |> display
    #state.individuals[491936] |> display
    event = pop!(state.queue)
    state.time = Simulation.time(event)
    #event |> display
    result = Simulation.execute!(state, params, event)
    #result |> println

end

Profile.clear_malloc_data()

@time for iter  in 0:10^4
    #println("iteration=$iter")
    if isempty(state.queue)
        @info "Empty queue after $iter events"
        break
    end
    #state.individuals[1] |> display
    #state.individuals[491936] |> display

    event = pop!(state.queue)
    
    state.time = Simulation.time(event)
    
    result = Simulation.execute!(state, params, event)
    #result |> println

end

