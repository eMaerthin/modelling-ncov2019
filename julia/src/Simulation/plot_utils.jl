using PyPlot
function plot_heatmap(results, title, tracking_probs = 0:0.05:1, Cs=0:0.05:1)
  figure(figsize=(10,5))
  reduction = (1 .- Cs) / 1.35  |> collect
  reduction |> println
  pcolor(reduction, tracking_probs, results', 
   norm=matplotlib.colors.LogNorm(vmin=minimum(results), vmax=maximum(results)),
   cmap="nipy_spectral")
  colorbar()

  xlabel("f - Stopień redukcji kontaktów")
  ylabel("b - Skuteczność wykrywania kontaktów")
 
  clim(vmin=10^2)
  gca().invert_yaxis()
  gca().invert_xaxis()
  savefig("tracking_heatmap_$title.png")
end