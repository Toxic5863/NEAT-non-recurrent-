isnumber(x) = isa(x, Number)
is_sensor(x) = getfield(x, :class) == 'S'
is_output(x) = getfield(x, :class) == 'O'
get_in(x) = getfield(x, :in)
is_active(x) = getfield(x, :activity) == true
get_historical_marker(x) = getfield(x, :innovation_number)


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
    if sigmoid(evaluate_row(x, y, z)) >= 0.5
        return true
    else
        return false
    end
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

    i += 1
    new_node = node_gene(node_count, 'H', i)
    x.nodes = vcat(x.nodes, new_node)

    i += 1
    new_connection = connection_gene(edge_in, node_count, 1, true, i)
    x.connections = vcat(x.connections, new_connection)

    i += 1
    new_connection = connection_gene(node_count, edge_out, edge_weight, true, i)
    x.connections = vcat(x.connections, new_connection)

    return i
end


function make_node_pair(x::genome)
    start_node = 0
    end_node = 0
    while start_node == end_node
        start_node = rand(1:length(x.nodes))
        end_node = rand(1:length(x.nodes))
    end
    return start_node, end_node
end


# the innovation count i must be set equal to the output of mutations
function add_edge(x::genome, i::Int)
    i += 1
    x_graphed = construct_network(x)
    non_recurrent_direction_found = false

    start_node, end_node = make_node_pair(x)
    # searching for a pair of nodes s.t. a cycle is not created via an edge between them
    while check_for_recurrent_connection(start_node, end_node, x_graphed)
        start_node, end_node = make_node_pair(x)
    end

    # creating the directed edge and adding it to the genome
    new_weight = rand() * rand(-8:8)
    new_connection = connection_gene(start_node, end_node, new_weight, true, i)
    x.connections = vcat(x.connections, new_connection)

    return i
end


# checks if node y is directly or indirectly an input for node x via adjacency matrix y
function check_for_recurrent_connection(node_x, node_y, y)
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
    modifier = rand(-6:6) * rand()
    x.connections[target_connection_index].weight *= modifier
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
    return [[generate_network(input_size) for a in 1:population_size]]
end


function mutate(a::genome, i::Int)
    mutation_chance = rand()
    if mutation_chance > 0.9
        mutation_choice = rand()
        if mutation_choice <= 0.95
            modify_weights(a)
        elseif mutation_choice <= 0.975
            i = add_node(a, i)
        else
            i = add_edge(a, i)
        end
    end
    return i
end


function get_genetic_range(x::genome)
    x_nodes_historical_min = minimum(get_historical_marker.(x.nodes))
    x_nodes_historical_max = maximum(get_historical_marker.(x.nodes))
    x_connections_historical_min = minimum(get_historical_marker.(x.connections))
    x_connections_historical_max = maximum(get_historical_marker.(x.connections))
    x_historical_minimum = minimum([x_nodes_historical_min x_connections_historical_min])
    x_historical_maximum = maximum([x_nodes_historical_max x_connections_historical_max])
    return x_historical_maximum - x_historical_minimum
end


function get_genetic_size(x::genome)
    return +(length(x.nodes), length(x.connections))
end


function compatibility(x::genome, y::genome)
    x_genetic_range, y_genetic_range = get_genetic_range(x), get_genetic_range(y)
    E = 0
    D = 0
    W = Float64[]
    x_genes = union(get_historical_marker.(x.nodes), get_historical_marker.(x.connections))
    y_genes = union(get_historical_marker.(y.nodes), get_historical_marker.(y.connections))
    for a ∈ x_genes
        if a ∉ y_genes
            if a > maximum(y_genes)
                E += 1
            else
                D += 1
            end
        end
    end
    for a ∈ y_genes
        if a ∉ x_genes
            if a > maximum(x_genes)
                E += 1
            else
                D += 1
            end
        end
    end
    for a ∈ intersect(get_historical_marker.(x.connections), get_historical_marker.(y.connections))
        for b ∈ y.connections
            if b.innovation_number == a
                for c ∈ x.connections
                    if c.innovation_number == a
                        push!(W, abs(b.weight - c.weight))
                    end
                end
            end
        end
    end

    W̅ = sum(W) / length(W)
    c₁ = 1
    c₂ = 1
    c₃ = 1
    x_size, y_size = get_genetic_size(x), get_genetic_size(y)
    N = max(x_size, y_size)
    σ = ((c₁ * D + c₂ * E) / N) + c₃ + W̅
    return σ
end


function crossover(x::genome, y::genome, x_fitness::Float64, y_fitness::Float64)
    matching_genes = Int[]

    if  x_fitness > y_fitness
        more_fit_parent = x
        less_fit_parent = y
    else
        more_fit_parent = y
        less_fit_parent = x
    end

    for a in x.nodes
        x_innovation_number = getfield(a, :innovation_number)
        for b in y.nodes
            y_innovation_number = getfield(b, :innovation_number)
            if x_innovation_number == y_innovation_number
                push!(matching_genes, x_innovation_number)
            end
        end
    end

    child_genome = deepcopy(more_fit_parent)

    for a in child_genome.connections
        if a.innovation_number in matching_genes
            if rand(0:1) > 0
                donor = more_fit_parent
            else
                donor = less_fit_parent
            end
            for b in donor.connections
                if a.innovation_number == b.innovation_number
                    a.weight = b.weight
                end
            end
        end
    end

    return child_genome
end


function f1score(x::genome, inputs::Array, outputs::Array)
    true_positives, false_positives = 0, 0
    true_negatives, false_negatives = 0, 0
    for input in 1:size(inputs)[1]
        if outputs[input, 1]
            if propagate(x, inputs[input, :])
                true_positives += 1
            else
                false_negatives += 1
            end
        else
            if propagate(x, inputs[input, :])
                false_positives += 1
            else
                true_negatives += 1
            end
        end
    end
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * ((precision * recall) / (precision + recall))
    println("inptus: ", inputs, " outputs: ", outputs, " f1: ", f1)
    return f1
end


function fitness(x::Array, inputs::Array, outputs::Array)
    solution_found = false
    solution = generate_network(1)
    errors = Any[]
    for species in 1:length(x)
        species_errors = Float64[]
        for genome in 1:length(x[species])
            push!(species_errors, f1score(x[species][genome], inputs, outputs))
        end
        push!(errors, species_errors)
    end
    return errors, solution_found, solution
end


function evolve_niche(niche::Array, errors::Array, i)
    roulette_wheel = errors
    roulette_wheel = roulette_wheel ./ sum(roulette_wheel)
    spin = rand()
    land = Int
    for a in 1:length(roulette_wheel)
        spin -= roulette_wheel[a]
        if spin <= 0
            land = a
        end
    end
    parent_1_index = land

    roulette_wheel = errors
    roulette_wheel = roulette_wheel ./ sum(roulette_wheel)
    spin = rand()
    land = Int
    for a in 1:length(roulette_wheel)
        spin -= roulette_wheel[a]
        if spin <= 0
            land = a
        end
    end
    parent_2_index = land

    child = crossover(niche[parent_1_index], niche[parent_2_index], errors[parent_1_index], errors[parent_2_index])

    for a in niche
        i = mutate(a, i)
    end

    child_placement = rand(1:length(niche))
    niche[child_placement] = child
    return i
end

# the species are represented as an array of arrays
# for determining compatibility, each species is represented by a random member
function evolve(speciated_genomes::Array, compatibility_threshold::Float64, target_conditions::Array, generations_left::Int, inputs::Array, outputs::Array, i)
    generations_left -= 1
    println("Generations left: ", generations_left)

    errors, solution_found, solution = fitness(speciated_genomes, inputs, outputs)
    for niche in 1:length(speciated_genomes)
        i = evolve_niche(speciated_genomes[niche], errors[niche], i)
    end
    if generations_left == 0
        target_conditions[2] = true
    end

    # re-speciation
    species_representatives = genome[]
    for a in speciated_genomes
        push!(species_representatives, a[rand(1:length(a))])
    end
    unspeciated_genomes = genome[]
    for a in speciated_genomes
        for b in a
            push!(unspeciated_genomes, b)
        end
    end
    next_generation = Any[]
    for a in species_representatives
        push!(next_generation, [a])
    end
    for a in unspeciated_genomes
        niche_found = false
        for b in next_generation
            species_representative = b[1]
            if compatibility(a, species_representative) < compatibility_threshold
                push!(b, a)
                niche_found = true
            end
        end
        if !niche_found
            push!(next_generation, [a])
        end
    end
    if solution_found
        target_conditions[1] = true
    end
    return next_generation, generations_left, i, solution
end


inputs = [0 0; 0 1; 1 0; 1 1]
outputs = [false; true; true; false]


function NEAT(inputs::Array, outputs::Array)
    σₜ = 1.6
    solution = generate_network(1)
    maximum_generations = 5000
    target_conditions = [false, false]
    # target conditions: minimum accuracy, maximum generations

    input_size = length(inputs[1])
    population = make_initial_population(30, input_size)
    i = input_size + 1
    while true ∉ target_conditions
        new_population, maximum_generations, i, solution = evolve(population, σₜ, target_conditions, maximum_generations, inputs, outputs, i)
    end

    # println("finished with ", maximum_generations, " generations left")
    println(solution)
    print(propagate(solution, inputs[3, :]))
end
