# This is a trial run of using the NEAT algorithm to train a neural network
import LinearAlgebra

# →(x, y) = getfield(x, y)    # custom operator for getting fields of structs


struct node_gene
    number::Int
    class::Char
end

struct connection_gene
    in::Int
    out::Int
    weight::Float64
    activity::Bool
    innovation_number::Int
end

struct genome
    nodes::Array
    connections::Array
end

function encode(x::Array, y::Array)
    return genome(x, y)
end

function sigmoid(x)
    return 1 / (1 + ℯ^-x)
end

is_sensor(x) = getfield(x, :class) == 'S'
is_active(x) = getfield(x, :activity) == true
get_in(x) = getfield(x, :in)

function propagate(x::genome, y::Array)
    input_layer = filter(is_sensor, x.nodes)
    active_connections = filter(is_active, x.connections)
    node_outputs = zeros(Float64, length(active_connections))
    sorted_active_connections = sort(active_connections, by=get_in)
end


# sample genome for debugging
a = genome([node_gene(1, 'S') node_gene(2, 'S') node_gene(3, 'O')], [connection_gene(1, 3, 1, true, 1) connection_gene(2, 3, 1, true, 1)])
