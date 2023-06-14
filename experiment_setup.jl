# # uncomment the following if not all packages are added.

# import Pkg
# using Pkg
# Pkg.add("HTTP")
# Pkg.add("JSON")
# Pkg.add("DataFrames")
# Pkg.add("OpenML")
# Pkg.add("DataFrames") 
# Pkg.add("CSV") 
# Pkg.add("Suppressor")
# Pkg.add("StatsBase")
# Pkg.add("ScikitLearn")
# Pkg.add("OpenSSL_jll")

using ScikitLearn
using ScikitLearn.Pipelines: Pipeline, FeatureUnion
using ScikitLearn.CrossValidation: cross_val_score
using XGBoost
using Revise
using Random
using Statistics: mean
using Suppressor
using Random
using Dates

include("lib/Herb.jl/src/Herb.jl")
include("lib/Herb.jl/HerbGrammar.jl/src/HerbGrammar.jl")
include("lib/Herb.jl/HerbData.jl/src/HerbData.jl")
include("lib/Herb.jl/HerbEvaluation.jl/src/HerbEvaluation.jl")
include("lib/Herb.jl/HerbConstraints.jl/src/HerbConstraints.jl")
include("lib/Herb.jl/HerbSearch.jl/src/HerbSearch.jl")
include("helper.jl")

# import the search algorithms
include("./search_algorithms/simple_enumerative_search.jl")
include("./search_algorithms/metropolis_hastings.jl")
include("./search_algorithms/vlns.jl")

# import the sk-learn functions
@sk_import decomposition: (PCA)
@sk_import preprocessing: (StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, Binarizer, PolynomialFeatures)
@sk_import feature_selection: (VarianceThreshold, SelectKBest, SelectPercentile, SelectFwe, RFE)
@sk_import tree: (DecisionTreeClassifier)
@sk_import ensemble: (RandomForestClassifier, GradientBoostingClassifier)
@sk_import linear_model: (LogisticRegression)
@sk_import neighbors: (NearestNeighbors, KNeighborsClassifier)
@sk_import svm: (LinearSVC)

Random.seed!(1234)

global pipelines_evaluated = 0

"""
Splits the dataset in the train set and test set. 
Also splits the features and labels.
"""
function load_and_split_dataset(dataset)
    
    # load and shuffle dataset
    dataset_shuffled = dataset[shuffle(1:end), :]
    
    # split into train and test sets 
    train_ind = floor(Int, size(dataset_shuffled, 1) * 0.75)
    val_ind = train_ind + floor(Int, size(dataset_shuffled, 1) * 0.15)
    train_df = dataset_shuffled[1:train_ind, :]
    val_df = dataset_shuffled[(train_ind+1):val_ind, :]
    test_df = dataset_shuffled[(val_ind+1):end, :]

    # split into features and labels
    train_X = train_df[:, 1:end-1]
    train_Y = train_df[:, end]
    test_X = test_df[:, 1:end-1]
    test_Y = test_df[:, end]
    val_X = val_df[:, 1:end-1]
    val_Y = val_df[:, end]

    return train_X, train_Y, val_X, val_Y, test_X, test_Y
end


# define the grammar
grammar = Herb.HerbGrammar.@cfgrammar begin
    
    # multiple classifiers possible
    START   = Pipeline([CLASSIF]) | Pipeline([PRE, CLASSIF])
    PRE     = PREPROC | FSELECT | ("seq", Pipeline([PRE, PRE]))  | ("par", FeatureUnion([BRANCH, BRANCH])) 
    BRANCH  = PRE | CLASSIF | ("seq", Pipeline([PRE, CLASSIF]))

    # preprocessing functions
    PREPROC =   
        ("StandardScaler", StandardScaler()) |
        ("RobustScaler", RobustScaler()) |
        ("MinMaxScaler", MinMaxScaler()) |
        ("MaxAbsScaler", MaxAbsScaler()) |
        ("PCA", PCA()) |
        ("Binarizer", Binarizer()) |
        ("PolynomialFeatures", PolynomialFeatures())

    # feature selection functions
    FSELECT =  
        ("VarianceThreshold", VarianceThreshold()) |
        ("SelectKBest",  SelectKBest(k=4)) |
        ("SelectPercentile",  SelectPercentile()) |
        ("SelectFwe",  SelectFwe()) |
        ("Recursive Feature Elimination",  RFE(LinearSVC())) 

    # classifiers
    CLASSIF =
        ("DecisionTree", DecisionTreeClassifier()) |
        ("RandomForest", RandomForestClassifier()) |
        ("Gradient Boosting Classifier", GradientBoostingClassifier()) |
        ("LogisticRegression", LogisticRegression()) #|
        ("KNearestNeighbors", KNeighborsClassifier())
end

"""
this method appends a number to each of the pipeline step names 
e.g. Pipeline(("op", op()), ("op", op()), ...) -> Pipeline(("op1", op()), ("op2", op()), ...)
this way all of the step names are unique
the input is the pipeline as a string, the output is also a string
"""
function insert_name_indexes(p)
    p_start = ""
    i = 1
    while i <= 100
        try
            p = replace(p, """",""" => string(i)*"""",""", count=1)
            p_split = split(p, string(i) * """", """)
            p_start *= p_split[1] * string(i) * """", """
            p = p_split[2]
            i += 1
        catch
            break
        end
    end
    return split(p_start, string(i))[1]
end

"""chooses a random pipeline from the grammar"""
function get_random_pipeline(grammar, max_depth, start_symbol)
    # all pipelines that can be assembled in max_depth steps
    cfe = Herb.HerbSearch.ContextFreeEnumerator(grammar, max_depth, start_symbol)
    cfe_size = deepcopy(cfe)
    
    # find size
    size = 0
    for pipeline in cfe_size
        size += 1
    end

    # Find start program
    ret_pipeline = nothing
    c = 0
    i = abs(rand(Int) % size)
    for pipeline in cfe
        if c == i
            ret_pipeline = pipeline
            break
        end
        c = c + 1
    end
    return ret_pipeline
end

"""chooses a random element from array"""
function rand_choose(a::Array)
    n = length(a)
    idx = mod(rand(Int64),n)+1
    return a[idx]
end



"""
Fits the pipeline to the training set and measures the accuracy on test set.
input:  pipeline, train_X, train_Y, test_X, test_Y
output: accuracy of pipeline
"""
function evaluate_pipeline(pipeline, train_X, train_Y, test_X, test_Y)
    global pipelines_evaluated += 1
    println(pipelines_evaluated)

    if pipelines_evaluated > max_pipelines
        return 0.0
    end

    # # this gives the following warning often, so it is suppressed for now.
    # # ConvergenceWarning: lbfgs failed to converge
    @suppress_err begin
        try
            # fit the pipeline
            model = ScikitLearn.fit!(pipeline, Matrix(train_X), Array(train_Y))

            # make predictions
            predictions = ScikitLearn.predict(model, Matrix(test_X))

            # measure the accuracy
            accuracy = mean(predictions .== test_Y)
            return accuracy
        catch e
            println("Caught error [in evaluate_pipeline()]: ", e)
            return 0.0
        end
    end
end

"""Trains the pipeline and returns 1-accuracy"""
function pipeline_cost_function(pipeline, train_X, train_Y, test_X, test_Y)
    return 1.0 - evaluate_pipeline(pipeline, train_X, train_Y, test_X, test_Y)
end

"""returns a string with information about the parameters"""
function parameters_info_string(search_alg_name, dataset_ids, n_runs, train_test_split, grammar, enumeration_depth, max_seconds, max_iterations_without_improvement, max_pipeline_depth, max_initial_pipeline_depth, neighbours_per_iteration)
    result  = 
        "Search algorithm: " * uppercase(search_alg_name) * "\n" *
        "Datasets: " * string(dataset_ids) * "\n" *
        "Number of runs: " * string(n_runs) * "\n" *
        "Start time: " * string(Dates.format(now(), "yyyy-mm-dd HH:MM:SS")) * "\n" * 

        "Parameters: " * "\n" * 
        "enumeration_depth: " * string(enumeration_depth) * "\n" * 
        "max_seconds: " * string(max_seconds) * "\n" * 
        "max_iterations_without_improvement: " * string(max_iterations_without_improvement) * "\n" * 
        "max_pipeline_depth: " * string(max_pipeline_depth) * "\n" * 
        "max_initial_pipeline_depth: " * string(max_initial_pipeline_depth) * "\n" *
        "neighbours_per_iteration (only vlns): " * string(neighbours_per_iteration) * "\n\n\n" *
        "Results: \n\n"

    return result
end

"""returns a string with information about te run"""
function run_info_string(dataset_id, grammar, search_alg_name, start_time, end_time, best_pipeline, best_cost, test_accuracy)
    duration = (end_time-start_time).value

    result  = 
        "Dataset: " * string(dataset_id) * "\n" *
        "Search algorithm: " * uppercase(search_alg_name) * "\n" *
        "Duration: " * string(round(duration/60000, digits=2)) * " minutes" *
        " (= " * string(round(duration/1000, digits=2)) * " seconds)" * "\n" * 
        "Best pipeline: " * string(best_pipeline) * "\n" * 
        "             : " * string(Herb.HerbSearch.rulenode2expr(best_pipeline, grammar)) * "\n" *
        "Cost: " * string(best_cost)* "\n" 
        "Test accuracy: " * string(test_accuracy) * "\n\n"
        
    return result
end

"""ruturns a string representation of a dictionary"""
function pretty_print_dict(d::Dict, pre=1)
    res = ""
    for (k,v) in d
        if typeof(v) <: Dict
            s = "$(repr(k)) => "
            res *= (join(fill(" ", pre)) * s) * "\n"
            res *= pretty_print_dict(v, pre+1+length(s)) * "\n"
        else
            res *= (join(fill(" ", pre)) * "$(repr(k)) => $(repr(v))") * "\n"
        end
    end
    return res
end

"""
Loads the datasets and runs the search algorithm n_runs amount of times.
It saves the results in a dictionary and returns this. 
"""
function run_search(
        search_alg_name, 
        dataset_ids,
        n_runs,
        train_test_split,
        grammar, 
        enumeration_depth, 
        max_seconds, 
        max_iterations_without_improvement, 
        max_pipeline_depth, 
        max_initial_pipeline_depth,
        neighbours_per_iteration)


    # information about the runs
    result_string = parameters_info_string(search_alg_name, dataset_ids, n_runs, train_test_split, grammar, enumeration_depth, max_seconds, max_iterations_without_improvement, max_pipeline_depth, max_initial_pipeline_depth, neighbours_per_iteration)
    results = Dict{Int, Any}()

    for dataset_id in dataset_ids
        # add new field to the results dict
        merge!(results, Dict(
            dataset_id => Dict(
                "cost" => [],
                "avg_cost" => 1.11,
                "sample_variance_cost" => 1.11,
                "test_accuracy" => [],
                "avg_test_accuracy" => 1.11,
                "sample_variance_test_accuracy" => 1.11
        )))
        ##load dataset HERE
        dataset = get_dataset(dataset_id)
        # run the search algorithm n_runs times
        for i in range(1, n_runs)
            # get train-test splits
            # Shuffle and split the dataset
            train_X, train_Y, val_X, val_Y, test_X, test_Y = load_and_split_dataset(dataset)
            data = [train_X, train_Y, val_X, val_Y]

            println(string(dataset_id) * " - " * string(i))
            best_program = nothing
            best_program_cost = nothing

            start_time = now()
            if search_alg_name == "mh"
                best_program, best_program_cost = mh(  grammar, data, enumeration_depth, max_seconds, max_iterations_without_improvement, max_pipeline_depth, max_initial_pipeline_depth)
            elseif search_alg_name == "vlns"
                best_program, best_program_cost = vlns(grammar, data, enumeration_depth, max_seconds, max_iterations_without_improvement, max_pipeline_depth, max_initial_pipeline_depth, neighbours_per_iteration)
            elseif search_alg_name == "bfs"
                best_program, best_program_cost = simple_enumerative_search(grammar, data, enumeration_depth, max_seconds)
            end
            end_time = now()
            
            pipeline = eval(Meta.parse(insert_name_indexes(string(Herb.HerbSearch.rulenode2expr(best_program, grammar)))))
            test_accuracy = evaluate_pipeline(pipeline, train_X, train_Y, test_X, test_Y)
            # add to result string
            result_string *= run_info_string(dataset_id, grammar, search_alg_name, start_time, end_time, best_program, best_program_cost, test_accuracy)

            # add to result dictionary
            push!(results[dataset_id]["cost"], best_program_cost)
            push!(results[dataset_id]["test_accuracy"], test_accuracy)
        end

        costs = results[dataset_id]["cost"]
        avg_cost = sum(costs)/n_runs
        results[dataset_id]["avg_cost"] = avg_cost
        results[dataset_id]["sample_variance_cost"] = sum((costs.-avg_cost).^2)/(n_runs-1)
        
        test_accuracies = results[dataset_id]["test_accuracy"]
        avg_test_accuracy = sum(test_accuracies)/n_runs
        results[dataset_id]["avg_test_accuracy"] = avg_test_accuracy
        results[dataset_id]["sample_variance_test_accuracy"] = sum((test_accuracies.-avg_test_accuracy).^2)/(n_runs-1)
    end

    result_string = pretty_print_dict(results) * "\n\n\n" * result_string
    return result_string, results
end

"""runs the search algorithm and saves the results to a file"""
function run_and_save(filename)
    #run algorithm - don't touch this!! (change at the top instead)
    result_string, results = run_search(search_alg_name, dataset_ids, n_runs, train_test_split, grammar, enumeration_depth, max_seconds, max_iterations_without_improvement, max_pipeline_depth, max_initial_pipeline_depth, neighbours_per_iteration)
    println(result_string)
    
    #write result to file
    open("db_output/"*string(filename)*".txt", "w") do file
        write(file, result_string)
    end

    return results
end

# clear output directory
foreach(rm, readdir("db_output",join=true))

### set the right parameters here

search_alg_name = "bfs"             # options: bfs, vlns, mh

dataset_ids = [61,1499]            # datasets: [seeds:1499, diabetes:37, tic-tac-toe:50, steel-plates-fault:1504]

n_runs = 2

max_pipelines = 10

# other parameters

# optimize this
enumeration_depth = 3
max_pipeline_depth = 5
neighbours_per_iteration = 15
max_iterations_without_improvement = 50


# don't optimize
train_test_split = 0.9
max_seconds = 3
max_initial_pipeline_depth = max_pipeline_depth


# save the avg costs in a new dict
final_results = Dict()
for id in dataset_ids
    final_results[id] = Dict("avg_costs" => [], "avg_variance_cost" => [], "avg_test_accuracy" => [], "avg_variance_test_accuracy" => [])
end



values = [3]  # set the values of the parameter you want to try out

# run the seach algorithms with the different values of the parameter
for value in values
    global enumeration_depth = value    # change the parameter name here
    res = run_and_save("enumeration_depth=" * string(value))

    # calculate the avg costs for each value
    for (id, val) in res
            push!(final_results[id]["avg_costs"], val["avg_cost"])
            push!(final_results[id]["avg_variance_cost"], val["sample_variance_cost"])
            push!(final_results[id]["avg_test_accuracy"], val["avg_test_accuracy"])
            push!(final_results[id]["avg_variance_test_accuracy"], val["sample_variance_test_accuracy"])
    end
end

# write avg costs to file 
open("db_output/avg_costs.txt", "w") do file
    write(file, pretty_print_dict(final_results))
end