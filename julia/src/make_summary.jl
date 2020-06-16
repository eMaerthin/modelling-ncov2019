using Distributed

@everywhere begin
  using CSV
  using DataFrames
  using FileIO
  using JLD2
  using ProgressMeter
  using Statistics
end

@assert length(ARGS) >= 1 "no experiment path given"

prefix_path = ARGS[1]
num_trajectories = length(ARGS)==2 ? parse(Int, ARGS[2]) : 1000

df = CSV.read(joinpath(prefix_path, "parameters_map.csv")) |> DataFrame;
  output_dirs = map( x-> joinpath(prefix_path, x, "output"), df.path)

results = @showprogress pmap(output_dirs) do path
  #println(stderr, "processing path $path")
  last_infection_times=Vector{Float32}()
  last_detection_times=Vector{Float32}()
  num_infections=Vector{UInt32}()
  num_detections=Vector{UInt32}()

  for trajectory in 1:num_trajectories
    try
      run_path = joinpath(path,"run_$trajectory.jld2")
      #println(stderr, "reading ", run_path)
      infection_times, detection_times = load(run_path, "infection_times", "detection_times")
      infection_times = filter!(!isnan, infection_times) |> sort!
      detection_times = filter!(!isnan, detection_times) |> sort!

      push!(last_infection_times, last(infection_times))
      push!(num_infections, length(infection_times))
      push!(last_detection_times, last(detection_times))
      push!(num_detections, length(detection_times))
    catch ex
      println(stderr, ex)
    end
  end
  return last_infection_times, last_detection_times, num_infections, num_detections
end

last_infections = getindex.(results, 1)
last_detections = getindex.(results, 2)
num_infections = getindex.(results, 3)
num_detections = getindex.(results, 4)

df.mean_last_infection_time = last_infections .|> mean
df.mean_last_detection_time = last_detections .|> mean
df.mean_num_infected = num_infections .|> mean
df.mean_num_detected = num_detections .|> mean

save(joinpath(prefix_path,"summary.jld2"),
  "summary", df,
  "last_infections", last_infections,
  "last_detections", last_detections,
  "num_infections", num_infections,
  "num_detections", num_detections
)

save(joinpath(prefix_path,"summary.csv"), df)