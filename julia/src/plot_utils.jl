using PyPlot

function plot_heatmap(results, title, tracking_probs = 0:0.05:1, Cs=0:0.05:1, logplot::Bool=true)
  figure(figsize=(10,5))
  reduction = 1 .- Cs / 1.35  |> collect
  if logplot
    pcolor(reduction, tracking_probs, results', 
     norm=matplotlib.colors.LogNorm(vmin=minimum(results), vmax=maximum(results)),
     cmap="nipy_spectral")
     clim(vmin=10^2)
  end     
  colorbar()

  xlabel("f - Stopień redukcji kontaktów")
  
  xticks(
    0.3:0.1:1,
    ["$(100*f)%" for f in 0.3:0.1:1])
  
  ylabel("b - Skuteczność wykrywania kontaktów")

  gca().invert_yaxis()
  gca().invert_xaxis()
  savefig("tracking_heatmap_$title.png")
end