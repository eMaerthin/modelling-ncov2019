
struct SimParams 
    household_ptrs::Vector{Tuple{Int32,Int32}}  # to be decided what it is - (list of pointers to the first member?)
    
    progressions::Vector{Progression}
    
    constant_kernel_param::Float64

end