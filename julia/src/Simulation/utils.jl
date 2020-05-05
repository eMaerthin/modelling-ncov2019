function countuniquesorted(arr::AbstractVector{T}) where T
    d = Dict{T,Int}()
    for val in arr
        count = get(d, val, 0)
        d[val] = count + 1
    end
    
    N = length(d)
    vals = Array{T,1}()
    sizehint!(vals,N)
    counts = Array{Int,1}()
    sizehint!(counts,N)
    for (val,count) in sort(collect(d))
        push!(vals,val)
        push!(counts,count)
    end
    
    return vals,counts
end

function groupptrs!(
  headptrs::AbstractVector{Ti}, 
  tailptrs::AbstractVector{Ti},
  arr::AbstractVector{T} where T<:Real, 
  ) where {Ti<:Integer}

  N = length(arr)
  @assert N == length(headptrs) == length(tailptrs)
  
  fill!(headptrs, 0)
  fill!(tailptrs, 0)
    
  headptr = 1
  flag = arr[1]
  headptrs[1] = headptr
    
  for i in 2:N
    @assert (arr[i-1] <= arr[i]) "array is not sorted at i=$i" #TODO this condition is not needed
    if flag != arr[i]
      flag = arr[i]
      headptr = i
    end  
    headptrs[i] = headptr
  end

  headptr = headptrs[end]
  tailptr = N
  
  tailptrs[end] = tailptr
  for i in (N-1):(-1):1
    if headptr != headptrs[i]
      headptr = headptrs[i]
      tailptr = i
    end 
    tailptrs[i] = tailptr
  end
     
  headptrs, tailptrs
end

groupptrs(arr::AbstractVector{T} where T<:Real) =
  groupptrs!(zeros(Int32, length(arr)), zeros(Int32, length(arr)), arr)
