import LinearAlgebra

isnumber(x) = isa(x, Number)
is_sensor(x) = getfield(x, :class) == 'S'
is_output(x) = getfield(x, :class) == 'O'
get_in(x) = getfield(x, :in)
is_active(x) = getfield(x, :activity) == true

mutable struct node_gene
    number::Int
    class::Char
    innovation_number::Int
end

mutable struct connection_gene
    in::Int
    out::Int
    weight::Float64
    activity::Bool
    innovation_number::Int
end

mutable struct genome
    nodes::Array
    connections::Array
end

function encode(x::Array, y::Array)
    return genome(x, y)
end

function sigmoid(x)
    return 1 / (1 + ℯ^-x)
end

function propagate(x::genome, y::Array)
    # the output node must be deteremined to be used as the starting point
    # for evaluating the output of the phenotype in a top-down manner
    output_node_number = filter(is_output, x.nodes)[1].number

    # using a directed adjacency matrix to model the phenotype
    # weighted edge i to j is represented as matrix[j, i] = weight
    node_count = length(x.nodes)
    weighted_adjacency_matrix = Array{Any}(undef, node_count, node_count)

    # setting the default entry to "e" for empty. this way, if an edge's
    # weight is not a number, we know to ignore it
    fill!(weighted_adjacency_matrix, 'e')

    # populating the adjacency matrix with weights
    for a in x.connections
        weighted_adjacency_matrix[a.out, a.in] = a.weight
    end

    # using top-down recursion to navigate the adjacency matrix
    # start at output node and navigate backwards along all edges with weights
    # until leaf nodes are arrived at. if leaves are sensor nodes, they return
    # the value of their corresponding inputs
    network_output = calculate_output(output_node_number, weighted_adjacency_matrix, y)
end

# w = weight of edge, x = origin vertex of edge, y = graph, z = inputs
function evaluate_connection(w, x, y, z)
    if isa(w, Number)
        return sigmoid(w * evaluate_row(x, y, z))
    else
        return 0
    end
end

# x = node number, y = adjacency matrix, z = inputs
function evaluate_row(x, y, z)
    if sum(isnumber.(y[x, :])) == 0
        if x <= length(z)
            return z[x]
        else
            return 0
        end
    else
        input_sum = 0
        for a in 1:length(y[x, :])
            input_sum += evaluate_connection(y[x, a], a, y, z)
        end
        return input_sum
    end
end

# x = node number, y = adjacency matrix, z = inputs
function calculate_output(x, y, z)
    return sigmoid(evaluate_row(x, y, z))
end

# the innovation count i must be set equal to the output of mutations
function add_node(x::genome, i::Int)
    found_active_connection = false
    target_connection_index = rand(1:length(x.connections))
    while found_active_connection == false
        if x.connections[target_connection_index].activity == true
            found_active_connection = true
            target_connection_index = rand(1:length(x.connections))
        end
    end
    node_count = length(x.nodes)
    node_count += 1

    x.connections[target_connection_index].activity = false
    edge_out = x.connections[target_connection_index].out
    edge_in = x.connections[target_connection_index].in
    edge_weight = x.connections[target_connection_index].weight

    new_node = node_gene(node_count, 'H', i)
    x.nodes = hcat(x.nodes, new_node)

    new_connection = connection_gene(edge_in, node_count, 1, true, i)
    x.connections = hcat(x.connections, new_connection)

    new_connection = connection_gene(node_count, edge_out, edge_weight, true, i)
    x.connections = hcat(x.connections, new_connection)
    return i
end

# the innovation count i must be set equal to the output of mutations
function add_edge(i)
end


i = 1
# sample genome and inputs for debugging
nodes = [node_gene(1, 'S', i) node_gene(2, 'H', i) node_gene(3, 'H', i) node_gene(4, 'O', i)]
connections = [connection_gene(1, 2, 1, true, 1) connection_gene(2, 3, 1, true, 2) connection_gene(3, 4, 1, true, 3)]
a = genome(nodes, connections)
input = [1]
# node 4 = w*sigmoid(sum(w*sigmoid()))
