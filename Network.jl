using Random

const Biases = Array{Array{Float64,1},1}
const Weights = Array{Array{Float64,2},1}
const Data = Tuple{Array{Float64,1},Array{Float64,1}}
const DataSet = Array{Data,1}

sigmoid(z) = 1 / (1 + exp(-z))
sigmoid_prime(z) = sigmoid(z) * (1 - sigmoid(z))

mutable struct Network
    biases::Biases
    weights::Weights
end

function Network(layers::Array{Int,1})::Network
    Network([randn(x) for x in layers[2:end]],
            [randn(x, y) for (x, y) in zip(layers[2:end], layers[1:end - 1])])
end

function train(network::Network, epochs::Int, batch_size::Int,
        learning_rate::Float64, training_data::DataSet, test_data::DataSet = nothing)
    for i in 1:epochs
        shuffle!(training_data)
        batches = [training_data[j:j + batch_size - 1]
                for j in 1:batch_size:length(training_data)]
        for batch in batches
            train_batch(network, batch, learning_rate)
        end
        if test_data == nothing
            println("Epoch $(i) complete")
        else
            println("Epoch $(i): $(evaluate(network, test_data)) / $(length(test_data))")
        end
    end
end

function train_batch(network::Network, batch::DataSet, learning_rate::Float64)
    nabla_b = [zeros(size(b)) for b in network.biases]
    nabla_w = [zeros(size(w)) for w in network.weights]
    for data in batch
        (delta_nabla_b, delta_nabla_w) = backpropagate(network, data)
        nabla_b = [nb + dnb for (nb, dnb) in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for (nw, dnw) in zip(nabla_w, delta_nabla_w)]
    end
    network.biases = [b - learning_rate * dbs / length(batch)
            for (b, dbs) in zip(network.biases, nabla_b)]
    network.weights = [w - learning_rate * dws / length(batch)
            for (w, dws) in zip(network.weights, nabla_w)]
end

function backpropagate(network::Network, (x, y)::Data)::Tuple{Biases,Weights}
    nabla_b = [zeros(size(b)) for b in network.biases]
    nabla_w = [zeros(size(w)) for w in network.weights]
    activations = [x]
    zs = []
    for (b, w) in zip(network.biases, network.weights)
        push!(zs, w * activations[end] .+ b)
        push!(activations, sigmoid.(zs[end]))
    end
    delta = (activations[end] .- y) .* sigmoid_prime.(zs[end])
    nabla_b[end] = delta
    nabla_w[end] = delta * transpose(activations[end - 1])
    for i in 1:length(zs) - 1
        z = zs[end - i]
        delta = transpose(network.weights[end - i + 1]) * delta .* sigmoid_prime.(z)
        nabla_b[end - i] = delta
        nabla_w[end - i] = delta * transpose(activations[end - i - 1])
    end
    (nabla_b, nabla_w)
end

function evaluate(network::Network, test_data::DataSet)::Int
    sum(argmax(feedforward(network, x)) == argmax(y) for (x, y) in test_data)
end

function feedforward(network::Network, input::Array{Float64,1})::Array{Float64,1}
    for (b, w) in zip(network.biases, network.weights)
        input = sigmoid.(w * input .+ b)
    end
    input
end