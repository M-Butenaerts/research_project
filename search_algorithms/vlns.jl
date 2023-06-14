"""constructs neighborhood given the pipeline and returns the best neighbor in its neighborhood"""
function find_best_neighbour_in_neighbourhood(current_program, grammar, data, enumeration_depth, t_start, max_time, max_pipeline_depth, neighbours_per_iteration, print_statements=false)
    # 1. Construct neighbourhood
    neighbourhood_node_loc, dict = Herb.HerbSearch.constructNeighbourhoodRuleSubset(current_program, grammar)
    replacement_expressions = Herb.HerbSearch.enumerate_neighbours_propose(enumeration_depth)(current_program, 
                                                                                                neighbourhood_node_loc, 
                                                                                                grammar,
                                                                                                max_pipeline_depth, # = max depth of pipeline, depth of subprogram is bound by this
                                                                                                dict)

    # 2. Find best neighbour
    best_program = deepcopy(current_program)
    pipeline = eval(Meta.parse(insert_name_indexes(string(Herb.HerbSearch.rulenode2expr(best_program, grammar)))))
    best_program_cost = pipeline_cost_function(pipeline, data[1], data[2], data[3], data[4])
    possible_program = current_program
    neighbours_tried = 0
    for replacement_expression in replacement_expressions
        t_now = now()
        if t_now - t_start > max_time
            if print_statements
                println("Timelimit reached")
            end
            break
        end
        if (neighbours_tried % 5) == 0
            if print_statements
                println("Tried ", neighbours_tried, " neighbours")
            end
        end
        if neighbours_tried == neighbours_per_iteration
            break
        end
        # change current_program to one of its neighbours 
        if neighbourhood_node_loc.i == 0
            possible_program = replacement_expression
        else
            neighbourhood_node_loc.parent.children[neighbourhood_node_loc.i] = replacement_expression
        end
        pipeline = eval(Meta.parse(insert_name_indexes(string(Herb.HerbSearch.rulenode2expr(possible_program, grammar)))))
        possible_program_cost = pipeline_cost_function(pipeline, data[1], data[2], data[3], data[4])
        if possible_program_cost <= best_program_cost
            best_program = deepcopy(current_program)
            best_program_cost = possible_program_cost        
        end
        neighbours_tried += 1
    end
    return best_program, best_program_cost
end


"""Very Large Neighborhood Search"""
function vlns(grammar, data, enumeration_depth, max_seconds, max_iterations_without_improvement, max_pipeline_depth, max_initial_pipeline_depth, neighbours_per_iteration=15, print_statements=false)
    current_program = get_random_pipeline(grammar, max_initial_pipeline_depth, :START)
    println("start pipeline: ", current_program)
    current_cost = 1.1
    i = 0
    not_improved_counter = 0
    max_time = Dates.Millisecond(max_seconds * 1000)
    t_start = now()
    while true
        t_now = now()
        if t_now - t_start > max_time
            break
        end
        if print_statements
            println("Iteration: ", i)
        end
        previous_cost = current_cost
        current_program, current_cost = find_best_neighbour_in_neighbourhood(current_program, grammar, data, enumeration_depth, t_start, max_time, max_pipeline_depth, neighbours_per_iteration, print_statements)

        if current_cost == previous_cost
            not_improved_counter += 1
            if not_improved_counter == max_iterations_without_improvement
                if print_statements
                    println("Stopping because hasn't inproved in ", max_iterations_without_improvement, " iterations")
                end
                break
            end
        else
            not_improved_counter = 0
        end

        if current_cost == 0.0
            if print_statements
                println("Stopping because cost is 0.0")
            end
            break
        end
        i += 1
    end
    if print_statements
        println("Final program: ", current_program)
        println("Final cost: ", current_cost)
    end
    return current_program, current_cost
end