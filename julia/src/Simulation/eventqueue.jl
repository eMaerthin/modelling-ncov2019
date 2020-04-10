using DataStructures

import DataStructures.compare

import Base: push!, pop!, isempty
export push!, pop!, isempty!

struct Earlier end
compare(c::Earlier, x::Event, y::Event) = time(x) == time(y) ? kind(x) < kind(y) : time(x) < time(y)


struct EventQueue
  immediates::CircularDeque{Event} 
  laters::BinaryHeap{Event, Earlier}
  EventQueue(immediate_size::Integer=128) = new(CircularDeque{Event}(immediate_size), BinaryHeap{Event, Earlier}())
end

push!(queue::EventQueue, event::Event; immediate::Bool=false) = begin immediate ? push!(queue.immediates, event) : push!(queue.laters, event); queue end
pop!(queue::EventQueue) = isempty(queue.immediates) ? pop!(queue.laters) : popfirst!(queue.immediates)
isempty(queue::EventQueue) = isempty(queue.immediates) & isempty(queue.laters)


