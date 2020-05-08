using DataFrames
using Distributions
using Random

function simple_run(individuals_df::DataFrame;
        tracking_prob::Float64=0.0, 
        constant_kernel_param::Float64=1.0, 
        tracking_delay::Float64=1.0,
        mild_detection_prob=0.0,
        history::Union{Nothing, Vector{Simulation.Event}}=nothing,
        execution_history::Union{Nothing, BitVector}=nothing,
        state_history::Union{Nothing, Vector{Simulation.IndividualState}}=nothing,
        debug_return=false,
        seed=123,
        initial_infections::Integer=100
    )
    rng = MersenneTwister(seed)
    params = Simulation.load_params(
        rng,
        population=individuals_df, 
        
        constant_kernel_param=constant_kernel_param,
        household_kernel_param=1.0,
        
        backward_tracking_prob=tracking_prob,
        backward_detection_delay=tracking_delay/2,
        
        forward_tracking_prob=tracking_prob,
        forward_detection_delay=tracking_delay/2,
        
        testing_time=tracking_delay/2,
        
        mild_detection_prob = mild_detection_prob
    );
    state = Simulation.SimState(params.progressions |> length, seed=seed)
    
    sample(rng, 1:length(params.progressions), initial_infections) .|> person_id -> 
        push!(state.queue, Simulation.Event(Val(Simulation.OutsideInfectionEvent), 0.0, person_id))

    @time Simulation.simulate!(
        state, 
        params, 
        history=history, 
        execution_history=execution_history, 
        state_history=state_history
    )
    
    return state
end


function simple_run!(state::Simulation.SimState, individuals_df::DataFrame;
        tracking_prob::Float64=0.0, 
        constant_kernel_param::Float64=1.0, 
        household_kernel_param::Real=0.3,

        tracking_delay::Float64=2.5,
        forward_detection_delay::Union{Nothing,Float64}=nothing,
        backward_detection_delay::Union{Nothing, Float64}=nothing,
        testing_time::Union{Nothing, Float64}=nothing,

        mild_detection_prob=0.0,
        phone_tracking_usage=0.0,
        phone_detection_delay=0.25,
        hospital_kernel_param::Float64=0.0,
        history::Union{Nothing, Vector{Simulation.Event}}=nothing,
        execution_history::Union{Nothing, BitVector}=nothing,
        state_history::Union{Nothing, Vector{Simulation.IndividualState}}=nothing,
        debug_return=false,
        seed=123,
        initial_infections::Integer=100
    )
    Simulation.reset!(state)
    
    state.rng = MersenneTwister(seed)
    
    

    params = Simulation.load_params(
        state.rng,
        population=individuals_df, 
        
        mild_detection_prob = mild_detection_prob,
        
        constant_kernel_param=constant_kernel_param,
        household_kernel_param=household_kernel_param,
        hospital_kernel_param=hospital_kernel_param,
        
        backward_tracking_prob=tracking_prob,
        backward_detection_delay= nothing==backward_detection_delay ? tracking_delay/2 : backward_detection_delay,
        
        forward_tracking_prob=tracking_prob,
        forward_detection_delay= nothing==forward_detection_delay ? tracking_delay/2 : forward_detection_delay,
        
        testing_time= nothing==testing_time ? tracking_delay/2 : testing_time,
        phone_tracking_usage=phone_tracking_usage,
        phone_detection_delay=phone_detection_delay
    );
    
    Simulation.initialfeed!(state, initial_infections)
 
    Simulation.simulate!(
        state, 
        params, 
        history=history, 
        execution_history=execution_history, 
        state_history=state_history
    )
    return state
end

function merge_history(history::Vector{Simulation.Event}, execution_history::BitVector, state_history::Vector{Simulation.IndividualState}; pop_last_event=false)
    history = copy(history)
    if pop_last_event
        last_event = pop!(history)
    end
    zip(history, state_history, execution_history)
end

function state2plot(state::Simulation.SimState)
    transmissions = vcat(state.forest.infections...)
    sort!(transmissions, lt=(x,y)->x.time<y.time)
    times = getproperty.(transmissions, :time)
    points = 1:length(transmissions)
    return times, points
end

