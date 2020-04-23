using PyPlot
using FileIO

mystep(r::AbstractRange) = step(r)
mystep(r::AbstractArray) = r[2] - r[1]

extendrange(r) = range( (first(r)-mystep(r)/2), last(r)+mystep(r)/2, length=length(r)+1)

function plot_heatmap(results, tracking_probs = 0:0.05:1, Cs=0:0.05:1; cmin=nothing, cmax=nothing, logscale=true, addcbar::Bool=true)
  figure(figsize=(10,5))
  reduction = 1 .- Cs / 1.35  |> collect  
  
  if nothing == cmax
    cmax = maximum(results)
  end   
  
  if nothing == cmin
    cmin = minimum(results)
  end 
  
  if logscale
    pcolor(
      extendrange(reduction), 
      extendrange(tracking_probs), 
      results', 
      norm=matplotlib.colors.LogNorm(vmin=cmin, vmax=cmax),
      cmap="nipy_spectral")
      clim(vmin=cmin)
  else
    pcolor(
      extendrange(reduction), 
      extendrange(tracking_probs), 
      results', 
      norm=matplotlib.colors.Normalize(vmin=cmin, vmax=cmax),
      cmap="nipy_spectral")
  end    
   
  if addcbar 
    cbar = colorbar()
  end

  xlabel("f - Stopień redukcji kontaktów")
  
  xticks(
    0:0.1:1,
    ["$(100*f)%" for f in 0:0.1:1])
  
  ylabel("b - skuteczność śledzenia kontaktów")

  gca().invert_yaxis()
  gca().invert_xaxis()
  #savefig("tracking_heatmap_$title.png", bbox_inches="tight")
end

function plot_heatmap_delay_vs_tracking_prob(
    data_path::AbstractString,
    image_path::AbstractString;
    addcbar::Bool=true,
    relative::Union{Nothing,Real} 
  )
#  figure(figsize=(10,5))
  
  data = load(data_path)
  
  map::AbstractArray{T,2} where T<:Real = data["map"]' 
  if relative!==nothing
    map /= relative
  end
  
  
  delays::AbstractVector{T} where T<:Real = data["delays"]
  tracking_probs::AbstractVector{T} where T<:Real = data["tracking_probs"]
  c::Real = data["c"]

  pcolor(
    extendrange(delays), 
    extendrange(tracking_probs), 
    map, 
    norm=matplotlib.colors.LogNorm(vmin=minimum(map), vmax=maximum(map)),
    cmap="nipy_spectral"
  )
  #clim(vmin=10^2)     

  if addcbar
    colorbar() 
  end
  #title("Łączna liczba zarażonych dla redukcji kontaktów o $(100*(1-c/1.35))%")
  xlabel("d - opóźnienie śledzenia w dniach")
  ylabel("b - skuteczność śledzenia kontaktów")

end

function plot_heatmap_mild_detection_vs_tracking_prob(
    data_path::AbstractString,
    image_path::AbstractString 
  )  
  data = load(data_path)
  
  map::AbstractArray{T,2} where T<:Real = data["map"]' 
  
  delay::Real = data["delay"]
  c::Real = data["c"]
  
  mild_detection_probs::AbstractVector{T} where T<:Real = data["mild_detection_probs"]
  tracking_probs::AbstractVector{T} where T<:Real = data["tracking_probs"]

  pcolor(
    extendrange(mild_detection_probs), 
    extendrange(tracking_probs), 
    map, 
    norm=matplotlib.colors.LogNorm(vmin=minimum(map), vmax=maximum(map)),
    cmap="nipy_spectral"
  )
  clim(vmin=10^2)     

  colorbar() 
  title("Łączna liczba zarażonych \n dla redukcji kontaktów o $(100*(1-c/1.35))% \n i opóźnienia śledzenia kontaktów o $delay dni")
  xlabel("prawdopodobieństwo wykrycia lekkich przypadków")
  ylabel("b - skuteczność śledzenia kontaktów")
  #xlim(minimum(mild_detection_probs), maximum(mild_detection_probs))
  #ylim(minimum(tracking_probs), maximum(tracking_probs))
  savefig(image_path, bbox_inches="tight")
end

function plot_heatmap_mild_detection_vs_tracking_prob(
    results, 
    mild_detections = 0:0.05:1, 
    tracking_probs=0:0.05:1; 
    cmin=nothing,
    cmax=nothing,
    logscale=true
  )  
  
  if nothing == cmax
    cmax = maximum(results)
  end   
  
  if nothing == cmin
    cmin = minimum(results)
  end 
  
  if logscale
    pcolor(
      extendrange(mild_detections), 
      extendrange(tracking_probs), 
      results', 
      norm=matplotlib.colors.LogNorm(vmin=cmin, vmax=cmax),
      cmap="nipy_spectral")
      clim(vmin=cmin)
  else
    pcolor(
      extendrange(mild_detections), 
      extendrange(tracking_probs), 
      results', 
      norm=matplotlib.colors.Normalize(vmin=cmin, vmax=cmax),
      cmap="nipy_spectral")
  end    
   
  clim(vmin=cmin)     

  colorbar() 
  #title("Łączna liczba zarażonych \n dla redukcji kontaktów o $(100*(1-c/1.35))% \n i opóźnienia śledzenia kontaktów o $delay dni")
  xlabel("prawdopodobieństwo wykrycia lekkich przypadków")
  ylabel("b - skuteczność śledzenia kontaktów")
  #xlim(minimum(mild_detection_probs), maximum(mild_detection_probs))
  #ylim(minimum(tracking_probs), maximum(tracking_probs))
  #savefig(image_path, bbox_inches="tight")
end

function plot_heatmap_mild_detection_vs_c(results, mild_detection_probss = 0:0.05:1, Cs=0:0.05:1; cmin=nothing, cmax=nothing, logscale=true, addcbar::Bool=true)
  figure(figsize=(10,5))
  reduction = 1 .- Cs / 1.35  |> collect
      
  if nothing == cmax
    cmax = maximum(results)
  end   
  
  if nothing == cmin
    cmin = minimum(results)
  end 
  
  if logscale
    pcolor(
      extendrange(reduction), 
      extendrange(mild_detection_probs), 
      results', 
      norm=matplotlib.colors.LogNorm(vmin=cmin, vmax=cmax),
      cmap="nipy_spectral")
      clim(vmin=cmin)
  else
    pcolor(
      extendrange(reduction), 
      extendrange(mild_detection_probs), 
      results', 
      norm=matplotlib.colors.Normalize(vmin=cmin, vmax=cmax),
      cmap="nipy_spectral")
  end    
  if addcbar
    c = colorbar()
  end
  xlabel("f - Stopień redukcji kontaktów")
  
  xticks(
    0:0.1:1,
    ["$(100*f)%" for f in 0:0.1:1])
  
  ylabel("q - Skuteczność wykrywania lekkich przypadków")

  gca().invert_yaxis()
  gca().invert_xaxis()
  #savefig("tracking_heatmap_$title.png", bbox_inches="tight")
end

function plot_heatmap_mild_detection_vs_c(
    data_path::AbstractString,
    image_path::AbstractString 
  )  
  data = load(data_path)
  
  map::AbstractArray{T,2} where T<:Real = data["map"]' 
  
  delay::Real = data["delay"]
  Cs::AbstractRange{T} where T<:Real = data["Cs"]
  mild_detection_probs::AbstractRange{T} where T<:Real = data["mild_detection_probs"]
  tracking_prob::Real = data["tracking_prob"]


  reduction = 1 .- Cs / 1.35  |> collect
  pcolormesh(
    extendrange(reduction), 
    extendrange(mild_detection_probs), 
#reduction,
#mild_detection_probs,
    map, 
    norm=matplotlib.colors.LogNorm(vmin=minimum(map), vmax=maximum(map)),
    cmap="nipy_spectral",
    shading="gouraud"
  )
  clim(vmin=10^2)     

  colorbar() 
  title("Łączna liczba zarażonych \n dla śledzenia kontaktów z prawdopodobieństwem b=$(tracking_prob*100)% i opóźnieniem $delay dni")
  
  xlabel("f stopień redukcji kontaktów")
  xticks(
    0.0:0.1:1,
    ["$(100*f)%" for f in 0.0:0.1:1])
  xticks(rotation=30)
  
  ylabel("prawdopodobieństwo wykrycia lekkich przypadków")
  #xlim(minimum(mild_detection_probs), maximum(mild_detection_probs))
  #ylim(minimum(tracking_probs), maximum(tracking_probs))
  savefig(image_path, bbox_inches="tight")
end
