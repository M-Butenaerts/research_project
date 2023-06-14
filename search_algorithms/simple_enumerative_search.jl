"""Simple Enumerative Search (BFS)"""
function simple_enumerative_search(grammar, data, enumeration_depth, max_seconds)

    # initialize values
    best_program = nothing
    best_cost = 1.1
    max_time = Dates.Millisecond(max_seconds * 1000)
    t_start = now()
        
    # enumerate pipelines in bfs manner
    enumerator = Herb.HerbSearch.ContextFreeEnumerator(grammar, enumeration_depth, :START)
    for rule in enumerator
        # check time
        t_now = now()
        if t_now - t_start > max_time
            break
        end

        if pipelines_evaluated > max_pipelines
            break
        end

        # evaluate the pipeline
        pipeline = eval(Meta.parse(insert_name_indexes(string(Herb.HerbSearch.rulenode2expr(rule, grammar)))))
        cost = pipeline_cost_function(pipeline, data[1], data[2], data[3], data[4])

        # update best cost
        if cost < best_cost
            best_cost = cost
            best_program = rule
        end
    end
    
    return best_program, best_cost
end
