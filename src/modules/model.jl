
using CUDA
using CUDA: CuArray, CuMatrix
import Base: copy

mutable struct LinearLayer
    #=
    This is an object that represent a 
    fully connected layer of a NN, it stores
    the layer parameters
    =#
    w::Matrix{Float32}
    b::Array{Float32, 1}
end

mutable struct CuLinearLayer
    w::CuMatrix{Float32}
    b::CuArray{Float32, 1}
end

function init_model(input_dim::Int, out_dim::Int, h_layers::Array{Int}, cuda::Bool)
    #=
    Returns a list of LinearLayers with paramaters
    initiaded from a normal distribution with μ = 0 and std = σ  
    =#
    if cuda
        model = CuLinearLayer[]
        σ = sqrt(1 / input_dim)
        push!(model, CuLinearLayer(CuMatrix{Float32}(randn(h_layers[1], input_dim) .* σ),
                                   CuArray{Float32,1}(zeros(h_layers[1]))))
        for i in 1:length(h_layers) - 1
            σ = sqrt(1 / h_layers[i])
            push!(model, CuLinearLayer(CuMatrix{Float32}(randn(h_layers[i + 1], h_layers[i]) .* σ),
                                       CuArray{Float32,1}(zeros(h_layers[i + 1]))))
        end
        σ = sqrt(1 / h_layers[end])
        push!(model, CuLinearLayer(CuMatrix{Float32}(randn(out_dim, h_layers[end]) .* σ),
                                   CuArray{Float32,1}(zeros(out_dim))))
        return model
    else
        model = LinearLayer[]
        σ = sqrt(1 / input_dim)
        push!(model, LinearLayer(Matrix{Float32}(randn(h_layers[1], input_dim) .* σ),
                                 Array{Float32,1}(zeros(h_layers[1]))))
        for i in 1:length(h_layers) - 1
            σ = sqrt(1 / h_layers[i])
            push!(model, LinearLayer(Matrix{Float32}(randn(h_layers[i + 1], h_layers[i]) .* σ),
                                     Array{Float32,1}(zeros(h_layers[i + 1]))))
        end
        σ = sqrt(1 / h_layers[end])
        push!(model, LinearLayer(Matrix{Float32}(randn(out_dim, h_layers[end]) .* σ),
                                 Array{Float32,1}(zeros(out_dim))))
        return model
    end
end

# function softmax(x::Vector)
#     exp.(x) ./ sum(exp.(x))
# end

function log_softmax(x::Vector)
    # fast implementation
    c = maximum(x)
    log_sum_exp = log(sum(exp.(x .- c)))
    x .- (c + log_sum_exp)
end

function cust_relu(x::Float32)
    # max(min(x, 1.), -1.) * 0.5 + 0.5
    max(min(x + 1f0, 1f0), 0f0)
end

function calc_allc(x::Vector)
    len_x = length(x)
    sum_x = sum(x)
    if sum_x > 0
        mean_x = mean(x)
        _x = x ./ sum_x
        _x = _x .* mean_x
        return [(1 - mean_x), _x...]
    end
    return [1., zeros(len_x)...]
end

function forward(model::Array{LinearLayer,1}, x::AbstractMatrix; arg_max::Bool=true)
    #=
    Run a forward pass of the input x through the model,
    the activation function is not applied to the output layer
    =#
    x = Array{Float32, 2}(x')
    for layer in model[1: end-1]
        x = layer.w * x .+ layer.b
        x = tanh.(x)
    end
    x = model[end].w * x .+ model[end].b
    x = x'

    if arg_max
        x = mapslices(log_softmax, x, dims=2)
        x = dropdims(mapslices(argmax, x, dims=2); dims=2)
    else
        x = cust_relu.(x)
        x = mapslices(calc_allc, x, dims=2)
        # x = mapslices(softmax, x, dims=2)
    end
    x
end

function forward(model::Array{CuLinearLayer,1}, x::AbstractMatrix; arg_max::Bool=true)
   #=
    Run a forward pass of the input x through the model,
    the activation function is not applied to the output layer
    =#
    x = CuArray{Float32, 2}(x')
    for layer in model[1: end-1]
        x = layer.w * x .+ layer.b
        x = CUDA.tanh.(x)
    end
    x = model[end].w * x .+ model[end].b
    x = Array{Float32, 2}(x')
    
    if arg_max
        x = mapslices(log_softmax, x, dims=2)
        x = dropdims(mapslices(argmax, x, dims=2); dims=2)
    else
        x = cust_relu.(x)
        x = mapslices(calc_allc, x, dims=2)
        # x = mapslices(softmax, x, dims=2)
    end
    x
end

# evolution

function get_exp_noise(model::AbstractArray, σ::Float64)
    #=
    Given a model it returns a list of NemdTuples,
    each tutple contains a 'w' and 'b' parameter 
    with the same size of their LinearLayer counterpart.
    They store the exploration noise that will be added
    to the original model to apply mutation
    =#
    exp_noise = NamedTuple[]
    for layer in model
        w_noise = Matrix{Float32}(randn(size(layer.w)) .* σ)
        b_noise = Array{Float32,1}(randn(size(layer.b)) .* σ)
        push!(exp_noise, (w = w_noise, b = b_noise))
    end
    exp_noise
end

function mutate(model::Array{LinearLayer,1}, exp_noise::Array{NamedTuple,1})
    #=
    It adds the exp_noise to each layer of the model
    and returns a new muteted model 
    =#
    mut_model = LinearLayer[]
    for i in eachindex(model)
        push!(mut_model, LinearLayer(model[i].w .+ exp_noise[i].w,
                                     model[i].b .+ exp_noise[i].b))
    end
    mut_model
end

function mutate(model::Array{CuLinearLayer,1}, exp_noise::Array{NamedTuple,1})
    mut_model = CuLinearLayer[]
    for i in eachindex(model)
        push!(mut_model, CuLinearLayer(model[i].w .+ CuMatrix{Float32}(exp_noise[i].w),
                                       model[i].b .+ CuArray{Float32,1}(exp_noise[i].b)))
    end
    mut_model
end

function compute_grad_approx(exp_noise::Array{Array{NamedTuple,1},1}, advantage::Array{Float32,1})
    #=
    Return the approximated gradient for each layer of the model,
    the gradient is approximated as the weighted sum between
    the exp_noise and its relative advantage
    =#
    grad = NamedTuple[]
    for i in 1:length(exp_noise[1])
        w_grad = sum([exp_noise[j][i].w .* advantage[j] for j in 1:length(advantage)])
        b_grad = sum([exp_noise[j][i].b .* advantage[j] for j in 1:length(advantage)])
        push!(grad, (w = w_grad, b = b_grad))
    end
    grad
end

function sgd_update!(model::Array{LinearLayer,1}, grad::Array{NamedTuple,1}, 
                     lr::Float64, σ::Float64, pop_size::Int, weight_decay::Float64)
    #=
    Using the approximated gradient apply one step of 
    Sstochastic Gradient Descent over the model parameters
    =#
    α = Float32(lr / (pop_size * σ))
    for i in 1:length(model)
        model[i].w = (model[i].w .* Float32(1 - weight_decay)) .- (grad[i].w .* α)
        model[i].b -= (grad[i].b .* α)
    end
end

function sgd_update!(model::Array{CuLinearLayer,1}, grad::Array{NamedTuple,1}, 
                     lr::Float64, σ::Float64, pop_size::Int, weight_decay::Float64)
    α = Float32(lr / (pop_size * σ))
    for i in 1:length(model)
        model[i].w = (model[i].w .* Float32(1 - weight_decay)) .- CuMatrix{Float32}(grad[i].w .* α)
        model[i].b -= CuArray{Float32,1}(grad[i].b .* α)
    end
end

# gpu

function copy(model::Array{LinearLayer,1})
    model_copy = Array{LinearLayer,1}()
    for layer in model
        layer_copy = LinearLayer(copy(layer.w),
                                 copy(layer.b))
        push!(model_copy, layer_copy)
    end
    model_copy
end

function copy(model::Array{CuLinearLayer,1})
    model_copy = Array{CuLinearLayer,1}()
    for layer in model
        layer_copy = CuLinearLayer(copy(layer.w),
                                   copy(layer.b))
        push!(model_copy, layer_copy)
    end
    model_copy
end

function to_cuda(model::Array{LinearLayer,1})
    model_cuda = Array{CuLinearLayer,1}()
    for layer in model
        layer_cuda = CuLinearLayer(convert(CuMatrix{Float32}, layer.w),
                                   convert(CuArray{Float32,1}, layer.b))
        push!(model_cuda, layer_cuda)
    end
    model_cuda
end

function to_cpu(model::Array{CuLinearLayer,1})
    model_cpu = Array{LinearLayer,1}()
    for layer in model
        layer_cpu = LinearLayer(convert(Matrix{Float32}, layer.w),
                                convert(Array{Float32,1}, layer.b))
        push!(model_cpu, layer_cpu)
    end
    model_cpu
end
