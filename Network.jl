struct Network
    biases::Array{Array{Float64,1},1}
    weights::Array{Array{Float64,2},1}
end

function Network(layers::Array{Int,1})::Network
    Network([randn(x) for x in layers[2:end]], [randn(x, y) for (x, y) in zip(layers[2:end], layers[1:end-1])])
end

function feedforward(network::Network, input::Array{Float64,1})::Array{Float64,1}
    a = input
    for (b, w) in zip(network.biases, network.weights)
        a = sigmoid.(w*a .+ b)
    end
    a
end

sigmoid(y) = 1 / (1 + exp(-y))

network = Network([2, 5, 2])

println(feedforward(network, [1., 1.]))