module Simulation
  using CSV
  using DataFrames
  using DataStructures
  using Distributions
  using GZip
  using JSON
  using LinearAlgebra
  using Logging
  using NPZ
  using ProgressMeter
  using PyPlot
  using PyCall

  using Distributions
  using Random

  import DataStructures.compare

  include("enums.jl")
  include("events.jl")

  struct Earlier end
  compare(c::Earlier, x::AbstractEvent, y::AbstractEvent) = time(x) < time(y)

  include("simstate.jl")
  include("progression.jl")
  include("simparams.jl")

  include("utils.jl")
  include("data_loading.jl")

  include("infection_kernels.jl")
end