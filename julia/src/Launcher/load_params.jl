using Random
using Distributions

function read_params(json, rng::AbstractRNG)
  constant_kernel_param = json["transmission_probabilities"]["constant"]  |> float
  household_kernel_param = json["transmission_probabilities"]["household"] |> float
  hospital_kernel_param = get(json["transmission_probabilities"], "hospital", 0.0) |> float
  friendship_kernel_param = get(json["transmission_probabilities"], "friendship", 0.0) |> float

  mild_detection_prob = json["detection_mild_proba"]  |> float

  tracking_prob = json["contact_tracking"]["probability"]  |> float
  tracking_backward_delay = json["contact_tracking"]["backward_detection_delay"]  |> float
  tracking_forward_delay = json["contact_tracking"]["forward_detection_delay"]  |> float
  testing_time = json["contact_tracking"]["testing_time"]  |> float

  phone_tracking = get(json, "phone_tracking", nothing)
  phone_tracking_usage = isnothing(phone_tracking) ? 0.0 : phone_tracking["usage"] |> float
  phone_tracking_testing_delay = isnothing(phone_tracking) ? 1.0 : phone_tracking : ["detection_delay"] |> float

  population_path = json["population_path"] # <= JSON
  population_path::AbstractString # checks if it was indeed a string

  individuals_df = load(population_path)["individuals_df"]

  infection_modulation_name, infection_modulation_params = if !haskey(json, "modulation")
    nothing, NamedTuple{}()
  else
    modulation = json["modulation"]
    params = get(modulation, "params", Dict{String,Any}())
    modulation["function"], NamedTuple{Tuple(Symbol.(keys(params)))}(values(params))
  end

  spreading = get(json, "spreading", nothing)
  spreading_alpha = isnothing(spreading) ? nothing : spreading["alpha"]
  spreading_x0 = isnothing(spreading) ? 1 : get(spreading, "x0", 1)
  spreading_truncation = isnothing(spreading) ? Inf : get(spreading, "truncation", Inf)

  Simulation.load_params(
    rng,
    population = individuals_df, 
        
    mild_detection_prob = mild_detection_prob,
        
    constant_kernel_param = constant_kernel_param,
    household_kernel_param = household_kernel_param,
    hospital_kernel_param = hospital_kernel_param,
    friendship_kernel_param = friendship_kernel_param,
        
    backward_tracking_prob = tracking_prob,
    backward_detection_delay = tracking_backward_delay,
        
    forward_tracking_prob = tracking_prob,
    forward_detection_delay = tracking_forward_delay,
        
    testing_time = testing_time,

    phone_tracking_usage = phone_tracking_usage,
    phone_detection_delay = phone_tracking_testing_delay,

    infection_modulation_name=infection_modulation_name,
    infection_modulation_params=infection_modulation_params,

    spreading_alpha=spreading_alpha,
    spreading_x0=spreading_x0,
    spreading_truncation=spreading_truncation
  )
end