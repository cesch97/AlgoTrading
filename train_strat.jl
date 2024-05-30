
using Pkg
Pkg.activate(".")

using YAML
using Distributed
addprocs(length(Sys.cpu_info()), exeflags="--project")

using ArgParse

include("src/strat_creation.jl")

function main()

    s = ArgParseSettings()
    @add_arg_table! s begin
        "conf_file"
            required = false
        "--shutdown", "-s"
            action = :store_true
        "--persist", "-p"
            action = :store_true
    end
    parsed_args = parse_args(s)
    shutdown = parsed_args["shutdown"]
    persist = parsed_args["persist"]

    if persist
        open("train_strat-persist.sh", "w") do io
            write(io, "#!/bin/bash\n")
            write(io, "JULIA_DEPOT_PATH=$(ENV["JULIA_DEPOT_PATH"])\n")
            write(io, "JULIA_NUM_THREADS=$(ENV["JULIA_NUM_THREADS"])\n")
            write(io, "OPENBLAS_NUM_THREADS=$(ENV["OPENBLAS_NUM_THREADS"])\n")
            write(io, "cd $(ENV["HOME"])/efs-1/AlgoTrading\n")
            screen_cmd = "screen -dm $(ENV["HOME"])/julia-1.5.0/julia ./train_strat.jl"
            (!isnothing(parsed_args["conf_file"])) && (screen_cmd *= " $(parsed_args["conf_file"])")
            (shutdown) && (screen_cmd *= " -s")
            write(io, screen_cmd * "\n")
        end
    end

    project_paths = YAML.load_file("./configs/paths.yml")
    if ENV["USER"] == "ec2-user"
        project_paths = project_paths["ec2"]
    else
        project_paths = project_paths["local"]
    end
    fx_data = YAML.load_file("./configs/fx_data.yml")

    if !isnothing(parsed_args["conf_file"])
        conf_file = parsed_args["conf_file"]
    else
        conf_file = YAML.load_file("./configs/strat_creation.yml")
        conf_file = merge(project_paths, fx_data, conf_file)
    end

    has_failed = false
    while true
        try
            strat_creation(conf_file)
            break
        catch e
            open("./logs/train_strat-log.txt", "a") do io
                write(io, "# ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), " #\n")
                e_msg = sprint(showerror, e, backtrace())
                write(io, e_msg, "\n\n")
            end
            (has_failed) && (break)
            has_failed = true
        end
    end

    rm("train_strat-persist.sh", force=true)

    if shutdown
        run(`sudo shutdown now -h`)
    end
end

main()