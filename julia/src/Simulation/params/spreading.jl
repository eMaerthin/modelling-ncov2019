using Distributions
using Random

struct SpreadingParams
  dist::Truncated{Pareto{Float64},Continuous,Float64}
  spreading::Vector{Float32}
end

function SpreadingParams(rng::AbstractRNG, N::Integer; alpha::Real, x0::Real=1, truncation::Real=Inf) 
  dist=truncated(Pareto(alpha, x0), x0, truncation)
  spreading = Vector{Float32}(undef, N)
  rand!(rng, dist, spreading)
  SpreadingParams(dist, spreading)
end

function saveparams(dict, p::SpreadingParams, prefix::AbstractString="")
  dict[prefix*"dist"] = params(p.dist)
  dict[prefix*"values"] = p.spreading
  nothing
end

function show(io::IO, p::SpreadingParams)
  print(io, "Spreading params for ", length(p), " individuals, distribution=", p.dist)
end

spreading(p::SpreadingParams, source_id::Integer) = Float(p.spreading[source_id])

