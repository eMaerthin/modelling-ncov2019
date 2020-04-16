struct InfectionForest
  sources::Vector{Event}
  infections::Vector{Vector{Event}}
  
  InfectionForest(num_individuals::Integer) = new(
    fill(Event(), num_individuals),
    [Vector{Event}() for i in 1:num_individuals]
  )
  
end

function reset!(forest::InfectionForest)
  @assert length(forest.sources) == length(forest.infections)
  
  fill!(forest.sources, Simulation.Event())
  
  for vec in forest.infections
    empty!(vec)
  end
  forest
end


function push!(forest::InfectionForest, infection::Event) 
  source_id = source(infection)

  if 0 == source_id
    @assert OutsideContact == contactkind(infection)
    return nothing
  end

  subject_id = subject(infection)
  
  @assert contactkind(forest.sources[subject_id]) == NoContact "The infection source should be assigned only once: $(forest.sources[subject_id])"
  forest.sources[subject_id] = infection
  
  push!(forest.infections[source_id], infection)
  
  nothing
end

forwardinfections(forest::InfectionForest, person_id::Integer)::Vector{Event} = forest.infections[person_id]
backwardinfection(forest::InfectionForest, person_id::Integer)::Event = forest.sources[person_id]