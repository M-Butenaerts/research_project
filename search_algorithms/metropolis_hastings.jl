"""Metropolis-Hastings search"""
function mh(grammar, data, enumeration_depth, max_seconds, max_iterations_without_improvement, max_pipeline_depth, max_initial_pipeline_depth)
    current_program = get_random_pipeline(grammar, max_initial_pipeline_depth, :START)
    current_accuracy = 0.0
    best_program = current_program
    best_accuracy = current_accuracy
    i = 0
    not_improved_counter = 0
    max_time = Dates.Millisecond(max_seconds * 1000)
    t_start = now()
    while true
        t_now = now()
        if t_now - t_start > max_time
            break
        end
        if (i % 10 == 0)
            println("iteration: ", i)
        end
        
        previous_accuracy = current_accuracy
        current_program, current_accuracy = one_iter_mh(current_program, grammar, data, enumeration_depth, t_start, max_time, max_pipeline_depth, current_accuracy)

        if current_accuracy == previous_accuracy
            not_improved_counter += 1
            if not_improved_counter == max_iterations_without_improvement
                println("stopping because hasn't inproved in" * string(max_iterations_without_improvement) * "iterations")
                break
            end
        else
            not_improved_counter = 0
        end
        if current_accuracy > best_accuracy
            best_accuracy = current_accuracy
            best_program = current_program
        end
        if current_accuracy == 1.0
            println("stopping because accuracy is 1.0")
            break
        end
        if pipelines_evaluated > max_pipelines
            break
        end
        i += 1
    end

    best_cost = 1.0 - best_accuracy

    println("final program: ", best_program)
    println("final cost: ", best_cost)
    return best_program, best_cost
end

"""Helper function for M-H search"""
function one_iter_mh(current_program, grammar, data, enumeration_depth, t_start, max_time, max_pipeline_depth, current_program_accuracy)
    # 1. Construct neighbourhood
    neighbourhood_node_loc, dict = Herb.HerbSearch.constructNeighbourhoodRuleSubset(current_program, grammar)
    replacement_expressions = Herb.HerbSearch.enumerate_neighbours_propose(enumeration_depth)(current_program, 
                                                                                                neighbourhood_node_loc, 
                                                                                                grammar,
                                                                                                max_pipeline_depth, # = max depth of pipeline, depth of subprogram is bound by this
                                                                                                dict)

    replacement_expression = rand_choose(replacement_expressions)

    original_program = deepcopy(current_program)  
    alternative_program = current_program                                                                                          
    if neighbourhood_node_loc.i == 0
        alternative_program = replacement_expression
    else
        neighbourhood_node_loc.parent.children[neighbourhood_node_loc.i] = replacement_expression
    end

    pipeline = eval(Meta.parse(insert_name_indexes(string(Herb.HerbSearch.rulenode2expr(alternative_program, grammar)))))
    alternative_program_cost = pipeline_cost_function(pipeline, data[1], data[2], data[3], data[4])
    alternative_program_accuracy = 1.0 - alternative_program_cost

    if (alternative_program_accuracy / current_program_accuracy) > ((rand(Int)%100000)/100000)
        return alternative_program, alternative_program_accuracy
    else
        return original_program, current_program_accuracy
    end
end
