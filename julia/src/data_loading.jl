using CSV
using GZip
using Distributions
using NPZ

include("utils.jl")

function load_individuals(path::AbstractString)::DataFrame
  df = GZip.open(path,"r") do io
    df = CSV.read(io, copycols=true)  # read CSV as DataFrame
    # create new DataFrame with
    DataFrame(
      age=Int8.(df.age),
      gender = df.gender .== 1,
      household_index = Int32.(df.household_index)
    )
  end
end

sample2dist(samples) = countuniquesorted(samples) |> x -> DiscreteNonParametric(x[1], x[2]./sum(x[2]) )
load_dist_from_samples(path::AbstractString) = npzread(path) |> sample2dist