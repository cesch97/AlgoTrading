
include("trad_sys_evo.jl")
using Hyperopt


function trad_sys_creation(config_file)
    if isa(config_file, String)
        config = YAML.load_file(config_file)
        log_dir = config["log_dir"]
    else
        config = config_file
        log_dir = config["trad_sys_dir"]
    end    
    log_name = config["log_name"]
    YAML.write_file("$log_dir/$log_name.yml", config)
    num_iter = config["hyp_opt-iter"]

    stat_ps = OrderedDict()
    rand_ps = OrderedDict()
    opt_ps = OrderedDict()
    for key in keys(config)
        (key == "hyp_opt-iter") && (continue)
        val = config[key]
        # key = Symbol(key)
        if isa(val, Array)
            if length(val) > 1
                if isa(val, Array{Float64,1})
                    opt_ps[key] = val
                else
                    rand_ps[key] = val
                end
            else
                stat_ps[key] = val[1]
            end
        else
            stat_ps[key] = val
        end
    end

    stat_ps = NamedTuple{Tuple(Symbol.(keys(stat_ps)))}(values(stat_ps))
    # rand_ps = NamedTuple{Tuple(Symbol.(keys(rand_ps)))}(values(rand_ps))
    opt_ps = NamedTuple{Tuple(Symbol.(keys(opt_ps)))}(values(opt_ps))

    checkpoint_file_name = "$log_dir/$log_name-checkpoint.jld2"
    i = 1
    if "$log_name-checkpoint.jld2" in readdir(log_dir, join=false)
        @load checkpoint_file_name i params candidates history results sampler
        ho = Hyperoptimizer(num_iter, params, candidates, history, results, sampler, nothing)
        i += 1
    else
        ho = Hyperoptimizer(num_iter, GPSampler(Max); opt_ps...)
    end

    strats_trad_data = Dict()

    iter_state = iterate(ho, i)
    while !isnothing(iter_state)
        samp_opt_ps, n_i = iter_state
        i = samp_opt_ps.i
        samp_opt_ps = (; [p for p in pairs(samp_opt_ps) if p[1] != :i]...) # remove "i"
        cfg_file_path = "$log_dir/configs/$log_name-$i.yml"
        if !(cfg_file_path in readdir("$log_dir/configs", join=true))
            samp_rand_ps = OrderedDict()
            for key in keys(rand_ps)
                samp_rand_ps[key] = rand(rand_ps[key])
            end
            samp_rand_ps = NamedTuple{Tuple(Symbol.(keys(samp_rand_ps)))}(values(samp_rand_ps))
            evo_ps = merge(stat_ps, samp_rand_ps, samp_opt_ps)
            evo_ps_dict = OrderedDict()
            for key in keys(evo_ps)
                evo_ps_dict[String(key)] = evo_ps[key]
            end
            evo_ps_dict["log_dir"] = "$log_dir/trad_systems"
            evo_ps_dict["log_name"] = "$log_name-$i"
            # delete!(evo_ps_dict, "i")
            YAML.write_file(cfg_file_path, evo_ps_dict)
        else
            opt_ps_dict = YAML.load_file(cfg_file_path)
            delete!.([opt_ps_dict,], String.(keys(stat_ps)))
            delete!.([opt_ps_dict,], keys(rand_ps))
            delete!(opt_ps_dict, "log_dir") # problem caused by removing the "log dir" in config files
            samp_opt_ps = NamedTuple{Tuple(Symbol.(keys(opt_ps_dict)))}(values(opt_ps_dict))
        end
        _, score, strats_trad_data = evolve_trad_sys(cfg_file_path, strats_trad_data)
        push!(ho.results, score)
        _samp_ps = []
        # for j in 2:length(samp_opt_ps)
        for j in eachindex(samp_opt_ps)
            push!(_samp_ps, samp_opt_ps[j])
        end
        # push!(ho.history, _samp_ps)
        ho.history[end] = _samp_ps
        begin
            params = ho.params
            candidates = ho.candidates
            history = ho.history
            results = ho.results
            sampler = ho.sampler
            @save checkpoint_file_name i params candidates history results sampler
        end
        println(" ")
        iter_state = iterate(ho, n_i)
    end
    rm("$log_name-checkpoint.jld2", force=true)
end
