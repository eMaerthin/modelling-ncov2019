struct PhoneTrackingParams
  isusingapp::BitVector
  detection_prob::Float64
  detection_delay::Float64
  
end

PhoneTrackingParams(N::Integer, detection_prob::Real=1.0, detection_delay::Real=0.5) = 
    PhoneTrackingParams(BitVector(undef, N), detection_prob, detection_delay)

function resample!(rng::AbstractRNG, params::PhoneTrackingParams, prob::Real)
    for i in 1:length(params.isusingapp)
        params.isusingapp[i] = rand(rng) < prob
    end
    params
end

PhoneTrackingParams(rng::AbstractRNG, N::Integer, usage::Real, detection_prob::Real=1.0, detection_delay::Real=0.5) =
    resample!(rng, PhoneTrackingParams(N, detection_prob, detection_delay), usage)

uses_phone_tracking(params::PhoneTrackingParams, person_id::Integer) = params.isusingapp[person_id]

