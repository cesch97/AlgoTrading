
using Pkg
Pkg.activate(".")

using YAML
using Distributed

include("src/trad_sys_evaluation.jl")
include("src/trad_sys_serving.jl")
include("src/strat_clean.jl")
include("src/trad_sys_clean.jl")
include("src/trad_sys_analysis.jl")

using ArgParse


function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "cmd"
            required = true
        "conf_file"
            required = false
    end
    parsed_args = parse_args(s)
    cmd = parsed_args["cmd"]

    project_paths = YAML.load_file("./configs/paths.yml")
    if ENV["USER"] == "ec2-user"
        project_paths = project_paths["ec2"]
    else
        project_paths = project_paths["local"]
    end
    fx_data = YAML.load_file("./configs/fx_data.yml")

    if cmd == "evaluate"
        if !isnothing(parsed_args["conf_file"])
            trad_sys_evaluation(parsed_args["conf_file"])
        else
            conf_file = YAML.load_file("./configs/trad_sys_evaluation.yml")
            conf_file = merge(project_paths, fx_data, conf_file)
            trad_sys_evaluation(conf_file)
        end
    elseif cmd == "serve"
        if !isnothing(parsed_args["conf_file"])
            trad_sys_serving(parsed_args["conf_file"])
        else
            conf_file = YAML.load_file("./configs/trad_sys_serving.yml")
            conf_file = merge(project_paths, fx_data, conf_file)
            trad_sys_serving(conf_file)
        end
    elseif cmd == "clean-strat"
        if !isnothing(parsed_args["conf_file"])
            clean_strats(parsed_args["conf_file"])
        else
            conf_file = YAML.load_file("./configs/strat_clean.yml")
            conf_file = merge(project_paths, fx_data, conf_file)
            clean_strats(conf_file)
        end
    elseif cmd == "clean-ts"
        if !isnothing(parsed_args["conf_file"])
            clean_trad_sys(parsed_args["conf_file"])
        else
            conf_file = YAML.load_file("./configs/trad_sys_clean.yml")
            conf_file = merge(project_paths, fx_data, conf_file)
            clean_trad_sys(conf_file)
        end
    elseif cmd == "analysis"
        if !isnothing(parsed_args["conf_file"])
            trad_sys_analysis(parsed_args["conf_file"])
        else
            conf_file = YAML.load_file("./configs/trad_sys_analysis.yml")
            conf_file = merge(project_paths, fx_data, conf_file)
            trad_sys_analysis(conf_file)
        end
    else
        throw(error("Not implemented!"))
    end
end

main()

