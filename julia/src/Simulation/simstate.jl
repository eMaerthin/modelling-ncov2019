


# the mutable part of the simulation

struct IndividualState #TODO change to immutable
  health::HealthState
  freedom::FreedomState
  detected::DetectionStatus
  quarantine_level::Int8 # allow for negative values to detect corruption
end

IndividualState() = IndividualState(
  Healthy,
  Free,
  Undetected,
  0
)

show(io::IO, s::IndividualState) = print(io, "(",s.health, ", ", s.freedom, ", ", s.detected, ", ", s.quarantine_level, ")")


mutable struct SimState
  rng::MersenneTwister

  time::TimePoint
  
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
    
  # buffers for rand
  sample_id_buf::Vector{UInt32}
  sample_time_buf::Vector{TimePoint}
    
    
  SimState(rng::AbstractRNG, num_individuals::Integer) = num_individuals <= 0 || num_individuals > typemax(UInt32) ? error("number of individuals must be positive and smaller than $(typemax(UInt32))") : 
    new(
      rng,
      
      0.0,
      #BinaryHeap{Event, Earlier}(),
      EventQueue(),
      
      [IndividualState() for i in  1:num_individuals],
      
      [Vector{Event}() for i in 1:num_individuals],
#      SortedMultiDict{UInt32, Event}(),

      fill((0x00, NoContact), num_individuals),
      
      0,
      0,
      0,
      
      # just the initial size, will be resized to meet the needs
      Vector{UInt32}(undef, 100),
      Vector{TimePoint}(undef, 100) 
    ) 
end

SimState(num_individuals::Integer; seed::Integer=0) = SimState(MersenneTwister(seed), num_individuals)

health(state::SimState, person_id::Integer)::HealthState = state.individuals[person_id].health
freedom(state::SimState, person_id::Integer)::FreedomState = state.individuals[person_id].freedom

quarantine_level(state::SimState, person_id::Integer) = state.individuals[person_id].quarantine_level
isquarantined(state::SimState, person_id::Integer)::Bool = quarantine_level(state, person_id) != 0
detected(state::SimState, person_id::Integer)::DetectionStatus = state.individuals[person_id].detected 
isdetected(state::SimState, person_id::Integer)::Bool = (Detected == detected(state, person_id))

subjecthealth(state::SimState, event::Event)::HealthState = health(state, subject(event))
subjectfreedom(state::SimState, event::Event)::FreedomState = freedom(state, subject(event))

sourcehealth(state::SimState, event::Event)::HealthState = health(state, source(event))
sourcefreedom(state::SimState, event::Event)::FreedomState = freedom(state, source(event))

#forwardinfections(state::SimState, person_id::Integer) = inclusive(state.infections, searchequalrange(state.infections, person_id)...) |> values
forwardinfections(state::SimState, person_id::Integer)::Vector{Event} = state.infections[person_id]
backwardinfection(state::SimState, person_id::Integer)::Tuple{UInt32,ContactKind} = state.infection_sources[person_id]


function sethealth!(state::SimState, person_id::Integer, new_health::HealthState)
  orig = state.individuals[person_id]
  state.individuals[person_id] = @set orig.health = new_health
  nothing
end

function setfreedom!(state::SimState, person_id::Integer, new_freedom::FreedomState)
  orig = state.individuals[person_id]
  state.individuals[person_id] = @set orig.freedom = new_freedom
  nothing
end

function setdetected!(state::SimState, person_id::Integer, new_detected::DetectionStatus)
  orig = state.individuals[person_id]
  state.individuals[person_id] = @set orig.detected = new_detected
  nothing
end

function quarantine_advance!(state::SimState, person_id::Integer, adv_val::Integer) 
  orig = state.individuals[person_id]
  state.individuals[person_id] = @set orig.quarantine_level = orig.quarantine_level+adv_val
  nothing
end

function quarantine_cancel!(state::SimState, person_id::Integer)
  orig = state.individuals[person_id]
  state.individuals[person_id] = @set orig.quarantine_level = 0 
  nothing
end

setsubjecthealth!(state::SimState, event::Event, health::HealthState) = sethealth!(state, subject(event), health)
setsubjectfreedom!(state::SimState, event::Event, freedom::FreedomState) = setfreedom!(state, subject(event), freedom)



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