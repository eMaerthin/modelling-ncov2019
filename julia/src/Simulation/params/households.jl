
function make_household_ptrs!(
    ptrs::AbstractVector{Tuple{Ti,Ti}},
    household_indices::AbstractVector{T} where T<:Integer
    ) where Ti<:Integer
  
    @assert length(ptrs) == length(household_indices)  
  
    ptrarr = reshape(reinterpret(Ti, ptrs), 2, :) # tuples as 2-by-N array
    headptrs = view(ptrarr, 1, :)
    tailptrs = view(ptrarr, 2, :)
    
    groupptrs!(headptrs, tailptrs, household_indices)
    ptrs
  end
  
  make_household_ptrs(household_indices::AbstractVector{T} where T<:Real) = 
    make_household_ptrs!(Vector{Tuple{Int32, Int32}}(undef, length(household_indices)), household_indices) 
  