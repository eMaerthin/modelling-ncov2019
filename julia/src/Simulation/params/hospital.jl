struct HospitalInfectionParams
    ishealthcare::BitVector
    hospital_staff_ids::Vector{UInt32}
    kernel_constant::Float64
    
    healthcare_detection_prob::Float64
    healthcare_detection_delay::Float64
  end

ishealthcare(params::HospitalInfectionParams, person_id::Integer) = params.ishealthcare[person_id]