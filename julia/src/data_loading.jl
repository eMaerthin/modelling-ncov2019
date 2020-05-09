using CSV
using DataFrames
using GZip
using Distributions
using NPZ
using FixedPointNumbers

function load_individuals(path::AbstractString)::DataFrame
  df = GZip.open(path,"r") do io
    df = CSV.read(io, copycols=true)  # read CSV as DataFrame
    # create new DataFrame with
    df = DataFrame(
      age=Int8.(df.age),
      gender = df.gender .== 1,
      household_index = Int32.(df.household_index),
      social_competence = Normed{UInt16,16}.(df.social_competence),
      ishealthcare = df.ishealthcare .== 1
    )
    sort!(df, :household_index)
  end
end

sample2dist(samples) = countuniquesorted(samples) |> x -> DiscreteNonParametric(x[1], x[2]./sum(x[2]) )
load_dist_from_samples(path::AbstractString) = npzread(path) |> sample2dist
