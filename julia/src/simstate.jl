# the mutable part of the simulation

mutable struct IndividualState
  health::HealthState
  freedom::FreedomState
  quarantine_level::Int8 # allow for negative values to detect corruption
  detected::Bool
  
  IndividualState() = new(
    Healthy,
    Free,
    0,
    false)
end

mutable struct SimState
  rng::MersenneTwister

  time::Float64
  queue::BinaryHeap{AbstractEvent, Earlier} # TODO change to union once all events are implemented
  
  individuals::Vector{IndividualState}  
      
  infections::SortedMultiDict{UInt32,AbstractInfectionEvent} # TODO change to union 
  infection_sources::Vector{UInt32}
    
  num_dead::Int
  num_affected::Int
  num_detected::Int
    
  SimState(num_individuals::Integer; seed=0) = num_individuals<=0 ? error("number of individuals must be positive") : 
    new(
      MersenneTwister(seed),
      
      0.0,
      BinaryHeap{AbstractEvent, Earlier}(),
      
      fill(IndividualState(), num_individuals),
      
      SortedMultiDict{Int32,AbstractInfectionEvent}(),
    
      0,
      0,
      0
    ) 
end


health(state::SimState, person_id::Integer) = state.individuals[person_id]
freedom(state::SimState, person_id::Integer) = state.individuals[person_id]

quarantine_advance!(state::SimState, person_id::Integer, val::Integer) = (state.individuals[person_id] += val)
quarantine_cancel!(state::SimState, person_id::Integer, val::Integer) = (state.individual[person_id] = 0)
isquarantined(state::SimState, person_id) = state.individuals[person_id].quarantine_level < 0 ? error("quarantine corrupted") : state.individuals[person_id] != 0
isdetected(state::SimState, person_id) = state.individuals[person_id].detected

subjecthealth(state::SimState, event::AbstractEvent) = health(state, subject(event))
subjectfreedom(state::SimState, event::AbstractEvent) = freedom(state, subject(event))

sourcehealth(state::SimState, event::TransmissionEvent) = health(state, source(event))
sourcefreedom(state::SimState, event::TransmissionEvent) = freedom(state, subject(event))

sethealth!(state::SimState, subject_id::Integer, health::HealthState) = (state.individuals[subject_id].health = health)
setfreedom!(state::SimState, subject_id::Integer, freedom::FreedomState) = (state.individual[subject_id].freedom = freedom)
setdetected!(state::SimState, subject_id::Integer, detected::Bool=true) = (state.individual[subject_id].detected = detected)

setsubjecthealth!(state::SimState, event::AbstractEvent, health::HealthState) = sethealth!(state, subject(event), health)
setsubjectfreedom!(state::SimState, event::AbstractEvent, freedom::FreedomState) = setfreedom!(state, subject(event), freedom)

infectiontargets(state::SimState, person_id::Integer) = inclusive(state.infections, searchequalrange(state.infections, person_id)...)
infectionsource(state::SimState, person_id::Integer)::Integer = state.infection_sources[person_id]
