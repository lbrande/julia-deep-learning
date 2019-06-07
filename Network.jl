struct Network
    biases::Array{Array{Float64,1},1}
    weights::Array{Array{Float64,2},1}
end

function Network(layers::Array{Int64,1})
    Network([randn(x) for x in layers[2:end]], [randn(x, y) for x in layers[2:end] for y in layers[1:end-1]])
end

network = Network([16, 20, 10])

println(size(network.biases))
println(size(network.weights))