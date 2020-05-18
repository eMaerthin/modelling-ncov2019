struct HospitalInfectionParams
    ishealthcare::BitVector
    hospital_staff_ids::Vector{UInt32}
    kernel_constant::Float64
    
    healthcare_detection_prob::Float64
    healthcare_detection_delay::Float64
  end

ishealthcare(params::HospitalInfectionParams, person_id::Integer) = params.ishealthcare[person_id]

function saveparams(dict, p::HospitalInfectionParams, prefix::AbstractString="")
  dict[prefix*"kernel_constant"] = p.kernel_constant
  dict[prefix*"detection_prob"] = p.healthcare_detection_prob
  dict[prefix*"healthcare_detection_delay"] = p.healthcare_detection_delay
end