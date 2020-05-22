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
  

#  if 0 == source_id
#    @assert 
#    return nothing
#  end

  subject_id = subject(infection)
  
  @assert contactkind(forest.inedges[subject_id]) == NoContact "The infection source should be assigned only once: $(forest.inedges[subject_id])"
  forest.inedges[subject_id] = infection

  source_id = source(infection)

  if OutsideContact != contactkind(infection)
    @assert 0 != source_id

    source_outdeg = forest.outdegrees[source_id] += 1
    key = EdgeDictKey(source_id, source_outdeg)
    @assert !haskey(forest.outedgedict, key)
    forest.outedgedict[key] = subject_id
  else
    @assert 0 == source_id
  end

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

function saveparams(dict, forest::RobinForest, prefix::AbstractString="")
  N = length(forest.inedges)
  infection_times = Vector{Float32}(undef, N)
  infection_sources = Vector{UInt32}(undef, N)
  contact_kinds = Vector{UInt8}(undef, N)
  @simd for i in 1:N
    event = forest.inedges[i]
    infected = NoContact != contactkind(event)
    infection_times[i] = infected ? Float32(time(event)) : NaN32
    infection_sources[i] = infected ? source(event) : 0
    contact_kinds[i] = contactkind(event) |> UInt8
  end
  dict[prefix*"infection_times"] = infection_times
  dict[prefix*"infection_sources"] = infection_sources
  dict[prefix*"contact_kinds"] = contact_kinds
end