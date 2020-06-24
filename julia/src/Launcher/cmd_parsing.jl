
using ArgParse

function parse_commandline()
	s = ArgParseSettings()
	@add_arg_table! s begin
		"JSON"
      help = "path to a JSON file with the parameter settings"
      required = true
      arg_type = String
    "--output-params-dump"
      help = "path where genereated simulation parameters (SimParams) are dumped"
      arg_type = String
		"--output-run-dump-prefix"
      help = "prefix that will be added to a filename were full output for each run should be written. Each subsequent trajectory i will be saved under  \$prefix_\$i.jld2"
      arg_type = String
    "--output-daily"
      help = "path where daily trajectories should be saved"
      arg_type = String
    "--output-summary"
      help = "path where summary should be saved"
      arg_type = String
	end
  parse_args(ARGS, s)
end