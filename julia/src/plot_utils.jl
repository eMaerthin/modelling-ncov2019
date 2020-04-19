using PyPlot
using FileIO

function plot_heatmap(results, title, tracking_probs = 0:0.05:1, Cs=0:0.05:1, logplot::Bool=true)
  figure(figsize=(10,5))
  reduction = 1 .- Cs / 1.35  |> collect
  if logplot
    pcolor(reduction, tracking_probs, results', 
     norm=matplotlib.colors.LogNorm(vmin=minimum(results), vmax=maximum(results)),
     cmap="nipy_spectral")
     clim(vmin=10^2)
  end     
  c = colorbar()

  xlabel("f - Stopień redukcji kontaktów")
  
  xticks(
    0.3:0.1:1,
    ["$(100*f)%" for f in 0.3:0.1:1])
  
  ylabel("b - Skuteczność wykrywania kontaktów")

  gca().invert_yaxis()
  gca().invert_xaxis()
  savefig("tracking_heatmap_$title.png")
end

function plot_heatmap_delay_vs_tracking_prob(
    data_path::AbstractString,
    title_str::Union{Nothing,AbstractString}, 
    image_path::AbstractString 
  )
#  figure(figsize=(10,5))
  
  data = load(data_path)
  
  map::AbstractArray{T,2} where T<:Real = data["map"]' 
  delays::AbstractVector{T} where T<:Real = data["delays"]
  tracking_probs::AbstractVector{T} where T<:Real = data["tracking_probs"]
  c::Real = data["c"]

  pcolor(delays, tracking_probs, map, 
    norm=matplotlib.colors.LogNorm(vmin=minimum(map), vmax=maximum(map)),
  
  cmap="nipy_spectral")
  clim(vmin=10^2)     

  colorbar() 
  title("Łączna liczba zarażonych dla c=$c (redukcja kontaktów o $(100*(1-c/1.35))%)")
  xlabel("opóźnienie śledzenia d")
  ylabel("skuteczność wykrywania kontaktów b")
  xlim(minimum(delays), maximum(delays))
  ylim(minimum(tracking_probs), maximum(tracking_probs))
  savefig(image_path)
end