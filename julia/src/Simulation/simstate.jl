# the mutable part of the simulation

mutable struct IndividualState #TODO change to immutable
  health::HealthState
  freedom::FreedomState
  quarantine_level::Int8 # allow for negative values to detect corruption
  detected::DetectionStatus
  
  IndividualState() = new(
    Healthy,
    Free,
    0,
    Undetected)
end


mutable struct SimState
  rng::MersenneTwister

  time::Float64
  
  #immediates::CircularDeque{Event} # immediate events
  #event_queue::BinaryHeap{Event, Earlier} # TODO change to union once all events are implemented
  queue::EventQueue
  
  individuals::Vector{IndividualState}  
  
  infections::Vector{Vector{Event}}    
  #infections::SortedMultiDict{UInt32,Event}
  
  infection_sources::Vector{Tuple{UInt32, ContactKind}}
  
    
  num_dead::Int
  num_affected::Int
  num_detected::Int
    
  SimState(num_individuals::Integer; seed=0) = num_individuals<=0 ? error("number of individuals must be positive") : 
    new(
      MersenneTwister(seed),
      
      0.0,
      #BinaryHeap{Event, Earlier}(),
      EventQueue(),
      
      [IndividualState() for i in  1:num_individuals],
      
      [Vector{Event}() for i in 1:num_individuals],
#      SortedMultiDict{UInt32, Event}(),

      fill((0x00, NoContact), num_individuals),
      
      0,
      0,
      0
    ) 
end


health(state::SimState, person_id::Integer)::HealthState = state.individuals[person_id].health
freedom(state::SimState, person_id::Integer)::FreedomState = state.individuals[person_id].freedom

quarantine_advance!(state::SimState, person_id::Integer, val::Integer) = (state.individuals[person_id].quarantine_level += val)
quarantine_cancel!(state::SimState, person_id::Integer) = (state.individuals[person_id].quarantine_level = 0)
quarantine_level(state::SimState, person_id::Integer)::Integer = state.individuals[person_id].quarantine_level
isquarantined(state::SimState, person_id::Integer)::Bool = quarantine_level(state, person_id) != 0
detected(state::SimState, person_id::Integer)::DetectionStatus = state.individuals[person_id].detected 
isdetected(state::SimState, person_id::Integer)::Bool = detected(state, person_id) == Detected

subjecthealth(state::SimState, event::Event)::HealthState = health(state, subject(event))
subjectfreedom(state::SimState, event::Event)::FreedomState = freedom(state, subject(event))

sourcehealth(state::SimState, event::Event)::HealthState = health(state, source(event))
sourcefreedom(state::SimState, event::Event)::FreedomState = freedom(state, subject(event))

sethealth!(state::SimState, subject_id::Integer, health::HealthState) = (state.individuals[subject_id].health = health)
setfreedom!(state::SimState, subject_id::Integer, freedom::FreedomState) = (state.individuals[subject_id].freedom = freedom)
setdetected!(state::SimState, subject_id::Integer, detected::DetectionStatus) = (state.individuals[subject_id].detected = detected)

setsubjecthealth!(state::SimState, event::Event, health::HealthState) = sethealth!(state, subject(event), health)
setsubjectfreedom!(state::SimState, event::Event, freedom::FreedomState) = setfreedom!(state, subject(event), freedom)

#forwardinfections(state::SimState, person_id::Integer) = inclusive(state.infections, searchequalrange(state.infections, person_id)...) |> values
forwardinfections(state::SimState, person_id::Integer)::Vector{Event} = state.infections[person_id]
backwardinfection(state::SimState, person_id::Integer)::Tuple{UInt32,ContactKind} = state.infection_sources[person_id]


function registerinfection!(state::SimState, infection::Event)
  source_id = source(infection)

  if 0 == source_id
    @assert OutsideContact == contactkind(infection)
    return nothing
  end

  subject_id = subject(infection)
  
  @assert state.infection_sources[subject_id][2] == NoContact "The infection source should be assigned only once: $(state.infection_sources[subject_id])"
  @inbounds state.infection_sources[subject_id] = (source_id, contactkind(infection))
  
  push!(state.infections[source_id], infection)
  
  nothing
end

#function registerinfection!(state::SimState, infection::Event)
#  println("ismissing") 
#  source_id = source(infection) 
  
#  if 0 == source_id
#    @assert OutsideContact == contactkind(infection)
#  end
    
  
#  println("subject")
#  subject_id = subject(infection)
  
#  println("source")
#  if 0 != source_id
#      state.infection_sources[subject_id] = (source_id, contactkind(infection))
#  end
  
#  println("push")
#  push!(state.infections, source_id => infection)
#  nothing
#end