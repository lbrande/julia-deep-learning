include("Network.jl")

function run(layers::Array{Int,1}, epochs::Int, batch_size::Int, learning_rate::Float64)
    train(Network(layers), epochs, batch_size, learning_rate, load("train"), load("t10k"))
end

function load(data_set_name::String)::DataSet
    image_stream = open("$(data_set_name)-images.idx3-ubyte")
    label_stream = open("$(data_set_name)-labels.idx1-ubyte")
    (image_count, image_size) = read_info(image_stream, label_stream)
    result = []
    for i in 1:image_count
        x = [convert(Float64, x) / 255 for x in read(image_stream, image_size)]
        y = one(zeros(10, 10))[read(label_stream, UInt8) + 1,:]
        push!(result, (x, y))
    end
    result
end

function read_info(image_stream::IOStream, label_stream::IOStream)::Tuple{Int,Int}
    throw(image_stream, 4)
    throw(label_stream, 8)
    (read_Int(image_stream), read_Int(image_stream) * read_Int(image_stream))
end

function read_Int(stream::IOStream)::Int
    result = 0
    for byte in read(stream, 4)
        result = (result << 8) + byte
    end
    result
end

function throw(stream::IOStream, nb::Int)
    read(stream, nb)
end