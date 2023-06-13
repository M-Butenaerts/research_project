
include("../../helper.jl")

using Statistics: mean
using ScikitLearn.CrossValidation: cross_val_score
using PyCall

using Random
using ScikitLearn
using ScikitLearn.Pipelines: Pipeline, FeatureUnion
using XGBoost
using Match
using Base
using InteractiveUtils

# using MLJ
# using MLJBase
# using MLJModels
@sk_import decomposition: (PCA)
@sk_import preprocessing: (StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, Binarizer, PolynomialFeatures)
@sk_import feature_selection: (VarianceThreshold, SelectKBest, SelectPercentile, SelectFwe, RFE)
@sk_import tree: (DecisionTreeClassifier)
@sk_import ensemble: (RandomForestClassifier, GradientBoostingClassifier)
@sk_import linear_model: (LogisticRegression)
@sk_import neighbors: (NearestNeighbors)
@sk_import neighbors: (KNeighborsClassifier)
@sk_import svm: (LinearSVC)

function fitness_function_function(dataset_nr)
    dataset = get_dataset(dataset_nr)

    dataset_shuffled = dataset[shuffle(1:end), :]
    # Calculate the split index
    split_index = floor(Int, size(dataset_shuffled, 1) * 0.9)

    # Split the data
    train_df = dataset_shuffled[1:split_index, :]
    test_df = dataset_shuffled[split_index+1:end, :]

    TRAIN_FEATURES = train_df[:, 1:end-1]
    TRAIN_LABELS = train_df[:, end]
    TEST_FEATURES = test_df[:, 1:end-1]
    TEST_LABELS = test_df[:, end]

    TRAIN_FEATURES = reshape(Matrix(TRAIN_FEATURES), size(TRAIN_FEATURES, 1), size(TRAIN_FEATURES, 2))
    TEST_FEATURES = reshape(Matrix(TEST_FEATURES), size(TEST_FEATURES, 1), size(TEST_FEATURES, 2))    
            
    function fitness_function(chromosome) 
        pipeline = parser(Herb.HerbSearch.rulenode2expr(chromosome, grammar), 0)[2]
        fitness = 0
        println("       PARSER DONE")
        try
            ScikitLearn.fit!(pipeline, TRAIN_FEATURES, String.(TRAIN_LABELS))
            println("       FIT DONE")
            predictions = ScikitLearn.predict(pipeline, TEST_FEATURES)
            println("       PREDICT DONE")
        
        fitness = mean(predictions .== TEST_LABELS)
        catch 
            println("   INVALID PIPELINE")
            fitness = 0
        end    
              
        println("       FITNESS: $(fitness)")
        return fitness
    end
    return fitness_function
end

function mutation(chromosome, grammar)
    # find all children 
    
    children = get_children(chromosome)
    # select random node
    (gen, loc) = children[rand(1:length(children))]
    
    loc = split(loc, ", ")[1:(length(split(loc, ", "))-1)]
    # find type 
    type = find_type(gen, grammar)
    # select random of same type
    new_gen = get_random_other_rule(gen, type, grammar)
    # exchange 
    exchange(chromosome, new_gen, loc)
    return chromosome
end

function get_children(chromosome)
    if (length(chromosome.children) == 0)
        return [(chromosome, "")]
    else
        children = []
        for child in chromosome.children
            for grand_child in get_children(child)
                push!(children, (grand_child[1], "$(child.ind), $(grand_child[2])")) 
            end
        end
        return children 
    end 
end


function find_type(gen, grammar)
    type = nothing
    for i in 1:length(grammar.types)
        if(Herb.HerbGrammar.rulenode2expr(gen, grammar) == grammar.rules[i])
            type = grammar.types[i]
        end
    end
    return type
end

function get_random_other_rule(gen, type, grammar)
    others = []
    
    for i in 1:length(grammar.types)
        if(grammar.types[i] == type && Herb.HerbGrammar.rulenode2expr(gen, grammar) != grammar.rules[i])
            new_gen = Herb.HerbGrammar.RuleNode(i)
            push!(others, new_gen)
        end
    end
    
    return others[rand(1:length(others))]
end

function exchange(chromosome, new_gen, loc)
    if (length(loc) == 1)
        for i in 1:length(chromosome.children)
            if ("$(chromosome.children[i].ind)" == loc[1])
                chromosome.children[i] = new_gen
                return 
            end
        end
    else 
        for i in 1:length(chromosome.children)
            if ("$(chromosome.children[i].ind)" == loc[1])
                exchange(chromosome.children[i], new_gen, loc[2:length(loc)])
            end
        end
    end

end


function stopping_condition(iteration, fitness)
    return (iteration > 10) || (fitness == 1.0)
end

function cross_over(chromosome1, chromosome2)
    
    # find crossing point 
    rules1 = get_cross_point(chromosome1)
    rules2 = get_cross_point(chromosome2)
    
    rule1 = rules1[rand(1:length(rules1)-1)]
    rule2 = rules2[rand(1:length(rules2)-1)]
    
    while length(get_cross_point(rule1)) != length(get_cross_point(rule2))
        rules1 = get_cross_point(chromosome1)
        rules2 = get_cross_point(chromosome2)
    
        rule1 = rules1[rand(1:length(rules1)-1)]
        rule2 = rules2[rand(1:length(rules2)-1)]
    end

    # println("rule1: $rule1")
    # println("rule2: $rule2")
    # println("")
    
    # cross over 
    exchange_rules(chromosome1, rule2, rule1)
    exchange_rules(chromosome2, rule1, rule2)    

    return (chromosome1, chromosome2) 
end

function get_cross_point(chromosome)
    rules = []
    for child in chromosome.children
        push!(rules, child)
        for grand_child in get_cross_point(child)
            push!(rules, grand_child)
        end 
    end
    return rules
end

function exchange_rules(chromosome, rule1, rule2)
    # println("chromosome: $chromosome")
    # println("rule1: $rule1")
    # println("rule2: $rule2")
    for i in (1:length(chromosome.children))
        
        if (chromosome.children[i] == rule2)
            chromosome.children[i] = rule1
            return
        end
        exchange_rules(chromosome.children[i], rule1, rule2) 
    end
    return
end

function parser(e::Expr, count)
    # println("$(e.args) \n\n")
    
    @match e.args begin 
        ["StandardScaler", :(StandardScaler())] => return ("$count", StandardScaler()) # GOOD
        ["RobustScaler", :(RobustScaler())] => return ("$count", RobustScaler())
        ["MinMaxScaler", :(MinMaxScaler())] => return ("$count", MinMaxScaler()) # GOOD
        ["MaxAbsScaler", :(MaxAbsScaler())] => return ("$count", MaxAbsScaler())
        ["PCA", :(PCA())] => return ("$count", PCA())
        ["Binarizer", :(Binarizer())] => return ("$count", Binarizer()) # GOOD
        ["PolynomialFeatures", :(PolynomialFeatures())] => return ("$count", PolynomialFeatures()) # GOOD
        
        ["VarianceThreshold", :(VarianceThreshold())] => return ("$count", VarianceThreshold()) # GOOD
        ["SelectKBest",  :(SelectKBest())] => return ("$count",  SelectKBest(k="all")) # GOOD
        ["SelectPercentile",  :(SelectPercentile())] => return ("$count",  SelectPercentile()) # GOOD
        ["SelectFwe",  :(SelectFwe())] => return ("$count",  SelectFwe()) # GOOD
        ["Recursive Feature Elimination",  :(RFE(LinearSVC()))] => return  ("$count",  RFE(LinearSVC(max_iter=10000))) # GOOD
    
        ["DecisionTree", :(DecisionTreeClassifier())] => return ("$count", DecisionTreeClassifier()) 
        ["RandomForest", :(RandomForestClassifier())] => return ("$count", RandomForestClassifier()) #GOOD
        ["Gradient Boosting Classifier", :(GradientBoostingClassifier())] => return ("$count", GradientBoostingClassifier()) 
        ["LogisticRegression", :(LogisticRegression())] => return ("$count", LogisticRegression()) # GOOD
        ["NearestNeighborClassifier", :(NearestNeighbors())] => return ("$count", KNeighborsClassifier(n_neighbors=5)) # GOOD
        
        [:Pipeline, x] => return ("$count", Pipeline([parser(x.args[1], "$count.1"), parser(x.args[2], "$count.2")]))
        [:FeatureUnion, x] => return ("$count", FeatureUnion([parser(x.args[1], "$count.1"), parser(x.args[2], "$count.2")]))
        
        ["seq", x] => return parser(x, "$count.1")
        ["par", x] => return parser(x, "$count.1")
        x => return -1
    end
end
