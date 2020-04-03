# the mutable part of the simulation

mutable struct IndividualState #TODO change to immutable
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
      
      [IndividualState() for i in  1:num_individuals],
      
      SortedMultiDict{UInt32,AbstractInfectionEvent}(),
      fill(0x00, num_individuals),
    
      0,
      0,
      0
    ) 
end


health(state::SimState, person_id::Integer)::HealthState = state.individuals[person_id].health
freedom(state::SimState, person_id::Integer)::FreedomState = state.individuals[person_id].freedom

quarantine_advance!(state::SimState, person_id::Integer, val::Integer) = (state.individuals[person_id].quarantine_level += val)
quarantine_cancel!(state::SimState, person_id::Integer) = (state.individuals[person_id].quarantine_level = 0)
quarantine_level(state::SimState, person_id::Integer)::Integer = state.individuals[person_id].quarantine_level < 0 ? error("quarantine corrupted") : state.individuals[person_id].quarantine_level
isquarantined(state::SimState, person_id)::Bool = quarantine_level(state, person_id) != 0
isdetected(state::SimState, person_id)::Bool = state.individuals[person_id].detected

subjecthealth(state::SimState, event::AbstractEvent)::HealthState = health(state, subject(event))
subjectfreedom(state::SimState, event::AbstractEvent)::FreedomState = freedom(state, subject(event))

sourcehealth(state::SimState, event::TransmissionEvent)::HealthState = health(state, source(event))
sourcefreedom(state::SimState, event::TransmissionEvent)::FreedomState = freedom(state, subject(event))

sethealth!(state::SimState, subject_id::Integer, health::HealthState) = (state.individuals[subject_id].health = health)
setfreedom!(state::SimState, subject_id::Integer, freedom::FreedomState) = (state.individuals[subject_id].freedom = freedom)
setdetected!(state::SimState, subject_id::Integer, detected::Bool=true) = (state.individuals[subject_id].detected = detected)

setsubjecthealth!(state::SimState, event::AbstractEvent, health::HealthState) = sethealth!(state, subject(event), health)
setsubjectfreedom!(state::SimState, event::AbstractEvent, freedom::FreedomState) = setfreedom!(state, subject(event), freedom)

forwardinfections(state::SimState, person_id::Integer) = inclusive(state.infections, searchequalrange(state.infections, person_id)...)
backwardinfection(state::SimState, person_id::Integer)::Integer = state.infection_sources[person_id]


function registerinfection!(state::SimState, infection::AbstractInfectionEvent) 
  source_id = ismissing(source(infection)) ? 0 : source(infection) 
  subject_id = subject(infection)
  
  state.infection_sources[subject_id] = source_id
  push!(state.infections, source_id => infection)
  nothing
end