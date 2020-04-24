import Base: iterate, length, eltype, sizehint!, push!

struct EdgeDictKey
  source_id::UInt32
  edge_id::UInt32
end

struct RobinForest
  inedges::Vector{Event}
  outdegrees::Vector{UInt32} 
  outedgedict::RobinDict{EdgeDictKey,UInt32}
  
end

RobinForest(sz::Integer) = RobinForest(fill(Event(), sz), fill(UInt32(0), sz), RobinDict{EdgeDictKey, UInt32}())


function sizehint!(forest::RobinForest, sz::Integer) 
    sizehint!(forest.inedges, sz)
    sizehint!(forest.outedges, sz)
    sizehint!(forest.outedgedict, sz)
end

function reset!(forest::RobinForest)
    fill!(forest.inedges, Event()),
    fill!(forest.outdegrees, 0),
    empty!(forest.outedgedict)
    fill!(forest.outedgedict.hashes, 0)
    nothing
end

function push!(forest::RobinForest, infection::Event) 
  source_id = source(infection)

  if 0 == source_id
    @assert OutsideContact == contactkind(infection)
    return nothing
  end

  subject_id = subject(infection)
  
  @assert contactkind(forest.inedges[subject_id]) == NoContact "The infection source should be assigned only once: $(forest.inedges[subject_id])"
  forest.inedges[subject_id] = infection
  
  source_outdeg = forest.outdegrees[source_id] += 1
  
  key = EdgeDictKey(source_id, source_outdeg)
  @assert !haskey(forest.outedgedict, key)
  forest.outedgedict[key] = subject_id
  
  nothing
end

backwardinfection(forest::RobinForest, person_id::Integer)::Event = forest.inedges[person_id]
forwardinfections(forest::RobinForest, person_id::Integer)::OutEdgeList = OutEdgeList(forest.outedgedict, forest.inedges, person_id, forest.outdegrees[person_id])

import Base.iterate
import Base.length
import Base.eltype

struct OutEdgeList
  dict::RobinDict{EdgeDictKey,UInt32}
  edges::Vector{Event}
  source_id::UInt32
  count::UInt32
end

iterate(oel::OutEdgeList, state=1) = state>oel.count ? nothing :
    (oel.edges[oel.dict[EdgeDictKey(oel.source_id, state)]], state+1) 
eltype(::Type{OutEdgeList}) = Event
length(oel::OutEdgeList) = oel.count