using Random

const DataSet = Array{Tuple{Array{Float64,1},Array{Float64,1}},1}

struct Network
    biases::Array{Array{Float64,1},1}
    weights::Array{Array{Float64,2},1}
end

function Network(layers::Array{Int,1})::Network
    Network([randn(x) for x in layers[2:end]], 
            [randn(x, y) for (x, y) in zip(layers[2:end], layers[1:end - 1])])
end

function feedforward(network::Network, input::Array{Float64,1})::Array{Float64,1}
    for (b, w) in zip(network.biases, network.weights)
        input = sigmoid.(w * input .+ b)
    end
    input
end

function train(network::Network, epochs::Int, batch_size::Int, 
    learning_rate::Float64, training_data::DataSet, test_data::DataSet = nothing)
    for i in 1:epochs
        shuffle!(training_data)
        batches = [training_data[j:j + batch_size - 1]
                for j in 1:batch_size:size(training_data)]
        for batch in batches
            train_batch(network, batch, learning_rate)
        end
        println(string("Epoch ", i, " complete"))
    end
end

function train_batch(network::Network, batch::DataSet, learning_rate::Float64)

end

sigmoid(z) = 1 / (1 + exp(-z))
sigmoid_prime(z) = sigmoid(z) * (1 - sigmoid(z))

network = Network([2, 5, 2])

println(feedforward(network, [1., 1.]))