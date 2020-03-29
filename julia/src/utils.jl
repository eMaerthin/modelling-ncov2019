function countuniquesorted(arr::AbstractArray{T,1}) where T
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