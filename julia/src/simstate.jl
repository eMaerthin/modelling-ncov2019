# the mutable part of the simulation
mutable struct SimState
  rng::MersenneTwister

  time::Float64
  queue::BinaryHeap{AbstractEvent, Earlier} # TODO change to union once all events are implemented
    
  health_states::Vector{HealthState}
  detected::BitVector
    
  infections::SortedMultiDict{Int32,AbstractInfectionEvent} # TODO change to union 
    
  num_dead::Int
  num_affected::Int
  num_detected::Int
    
  SimState(num_individuals::Integer; seed=0) = num_individuals<=0 ? error("number of individuals must be positive") : 
    new(
      MersenneTwister(seed),
      
      0.0,
      BinaryHeap{AbstractEvent, Earlier}(),
      
      fill(Healthy, num_individuals),
      falses(num_individuals),
      
      SortedMultiDict{Int32,AbstractInfectionEvent}(),
    
      0,
      0,
      0
    ) 
end


health(state::SimState, person_id::Integer) = state.health_states[person_id]

subjecthealth(state::SimState, event::AbstractEvent) = health(state, subject(event))
sourcehealth(state::SimState, event::TransmissionEvent) = health(state, source(event))

sethealth!(state::SimState, subject_id::Integer, health::HealthState) = (state.health_states[subject_id] = health)
setsubjecthealth!(state::SimState, event::AbstractEvent, health::HealthState) = sethealth!(state, subject(event), health)
