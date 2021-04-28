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

function construct_network(x::genome)
    # using a directed adjacency matrix to model the phenotype
    # weighted edge i to j is represented as matrix[j, i] = weight
    node_count = length(x.nodes)
    weighted_adjacency_matrix = Array{Any}(undef, node_count, node_count)

    # setting the default entry to "e" for empty. this way, if an edge's
    # weight is not a number, we know to ignore it.
    fill!(weighted_adjacency_matrix, 'e')

    # populating the adjacency matrix with weights
    for a in x.connections
        weighted_adjacency_matrix[a.out, a.in] = a.weight
    end

    return weighted_adjacency_matrix
end

function propagate(x::genome, y::Array)
    phenotype = construct_network(x)

    # the output node must be deteremined to be used as the starting point
    # for evaluating the output of the phenotype in a top-down manner
    output_node_number = filter(is_output, x.nodes)[1].number

    # using top-down recursion to navigate the adjacency matrix
    # start at output node and navigate backwards along all edges with weights
    # until leaf nodes are arrived at. if leaves are sensor nodes, they return
    # the value of their corresponding inputs
    network_output = calculate_output(output_node_number, phenotype, y)
end

# w = weight of edge, x = origin vertex of edge, y = graph, z = inputs
# function evaluate_connection(w, x, y, z)
#     if isa(w, Number)
#         return sigmoid(w * evaluate_row(x, y, z))
#     else
#         return 0
#     end
# end

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
            if isa(y[x, a], Number)
                input_sum += sigmoid(y[x, a] * evaluate_row(a, y, z))
            end
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
        else
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
    x.nodes = vcat(x.nodes, new_node)

    new_connection = connection_gene(edge_in, node_count, 1, true, i)
    x.connections = vcat(x.connections, new_connection)

    new_connection = connection_gene(node_count, edge_out, edge_weight, true, i)
    x.connections = vcat(x.connections, new_connection)
end

function make_node_pair(x::genome)
    start_node = 0
    end_node = 0
    while start_node == end_node
        print("redoing it")
        start_node = rand(1:length(x.nodes))
        end_node = rand(1:length(x.nodes))
    end
    return start_node, end_node
end

# the innovation count i must be set equal to the output of mutations
function add_edge(x::genome, i::Int)
    x_graphed = construct_network(x)
    non_recurrent_direction_found = false

    start_node, end_node = make_node_pair(x)
    # searching for a pair of nodes s.t. a cycle is not created via an edge between them
    while check_for_recurrent_connection(start_node, end_node, x_graphed)
        start_node, end_node = make_node_pair(x)
    end

    # creating the directed edge and adding it to the genome
    new_connection = connection_gene(start_node, end_node, 1, true, i)
    x.connections = vcat(x.connections, new_connection)
end

# checks if node y is directly or indirectly an input for node x via adjacency matrix y
function check_for_recurrent_connection(node_x, node_y, y)
    print(y[node_x, :])
    if isnumber(y[node_x, node_y])
        return true
    else
        for a in 1:length(y[node_x, :])
            if isnumber(y[node_x, a])
                if check_for_recurrent_connection(a, node_y, y)
                    return true
                end
            end
        end
    end
    return false
end

function modify_weights(x::genome)
    target_connection_index = rand(1:length(x.connections))
    x.connections[target_connection_index].weight *= rand(-3:3)
end

# x = inputs
function generate_network(input_size::Int)
    nodes = Array{node_gene}(undef, input_size+1)
    connections = Array{connection_gene}(undef, input_size)
    for a in 1:input_size
        nodes[a] = node_gene(a, 'S', a)
    end
    output_node_number = length(nodes)
    nodes[output_node_number] = node_gene(output_node_number, 'O', output_node_number)
    for a in 1:input_size
        connections[a] = connection_gene(a, output_node_number, 1, true, a)
    end
    return genome(nodes, connections)
end

function make_initial_population(population_size::Int, input_size::Int)
    return [generate_network(input_size) for a in 1:population_size]
end

function fitness(x::genome, inputs::Array, outputs::Array)
    error = Array{Float64}(undef, length(outputs))
    for a in 1:length(outputs)
        error[a] = outputs[a] - propagate(x, inputs[a, :])
    end
    error = abs.(error)
    return sum(error)/length(error)
end

function selection(population::Array, inputs::Array, outputs::Array)
    roulette_wheel = Array{Float64}(undef, length(population))
    for a in 1:length(population)
        roulette_wheel[a] = fitness(population[a], inputs, outputs)
    end
    roulette_wheel = roulette_wheel ./ sum(roulette_wheel)
    parents = Array{genome}(undef, 2)
    parent_seed = rand()
    for a in 1:length(roulette_wheel)
        parent_seed -= roulette_wheel[a]
        if parent_seed <= 0
            parents[1] = population[a]
        end
    end
    parent_seed = rand()
    for a in 1:length(roulette_wheel)
        parent_seed -= roulette_wheel[a]
        if parent_seed <= 0
            parents[2] = population[a]
        end
    end
    return(parents)
end

function mutation(parents::Array, inputs::Array, outputs::Array, i::Int)
    mutation_chance = rand()
    for a in parents
        if mutation_chance > 0.1
            i += 1
            mutation_choice = rand()
            if mutation_choice <= 0.6
                modify_weights(a)
            elseif mutation_choice <= 0.8
                add_node(a, i)
            else
                add_edge(a, i)
            end
        end
    end
    return i
end

function compatability(x::genome, y::genome)
    σ = "f"
end

inputs = [0 0; 0 1; 1 0; 1 1]
outputs = [0; 1; 1; 0]

function NEAT(inputs::Array, outputs::Array)
    i = 1
    input_size = length(inputs[1])
    population = make_initial_population(100, input_size)
    parents = selection(population, inputs, outputs)
    parents = mutation(parents, inputs, outputs, i)
end

# sample genome and inputs for debugging
# nodes = [node_gene(1, 'S', i) node_gene(2, 'H', i) node_gene(3, 'H', i) node_gene(4, 'O', i)]
# connections = [connection_gene(1, 2, 1, true, 1) connection_gene(2, 3, 1, true, 2) connection_gene(3, 4, 1, true, 3)]
# a = genome(nodes, connections)
# input = [1]
# node 4 = w*sigmoid(sum(w*sigmoid()))
