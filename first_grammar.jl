
using ScikitLearn
using ScikitLearn.Pipelines: Pipeline, FeatureUnion
using XGBoost
using Revise

include("lib/Herb.jl/src/Herb.jl")
include("genetic_algorithm/src/genetic_algorithm.jl")

@sk_import decomposition: (PCA)
@sk_import preprocessing: (StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, Binarizer, PolynomialFeatures)
@sk_import feature_selection: (VarianceThreshold, SelectKBest, SelectPercentile, SelectFwe, RFE)
@sk_import tree: (DecisionTreeClassifier)
@sk_import ensemble: (RandomForestClassifier, GradientBoostingClassifier)
@sk_import linear_model: (LogisticRegression)
@sk_import neighbors: (NearestNeighbors)
@sk_import svm: (LinearSVC)

sequence(a, b) = Pipeline([a, b]) 
parallel(a, b) = FeatureUnion([a, b]) 

grammar = Herb.HerbGrammar.@cfgrammar begin
    # START   = CLASSIF | sequence(PRE, CLASSIF)
    # PRE     = PREPROC | FSELECT | sequence(PRE, PRE) | parallel(BRANCH, BRANCH)
    # BRANCH  = PRE | CLASSIF | sequence(PRE, CLASSIF) 

    
    START   = Pipeline([CLASSIF]) | Pipeline([PRE, CLASSIF])
    PRE     = PREPROC | FSELECT | ("seq", Pipeline([PRE, PRE]))  | ("par", FeatureUnion([PRE, PRE])) 
    # BRANCH  = PRE | CLASSIF | Pipeline([PRE, CLASSIF]) 

    PREPROC =   
        ("StandardScaler", StandardScaler()) |
        ("RobustScaler", RobustScaler()) |
        ("MinMaxScaler", MinMaxScaler()) |
        ("MaxAbsScaler", MaxAbsScaler()) |
        ("PCA", PCA()) |
        ("Binarizer", Binarizer()) |
        ("PolynomialFeatures", PolynomialFeatures())
    FSELECT =  
        ("VarianceThreshold", VarianceThreshold()) |
        ("SelectKBest",  SelectKBest()) |
        ("SelectPercentile",  SelectPercentile()) |
        ("SelectFwe",  SelectFwe()) |
        ("Recursive Feature Elimination",  RFE(LinearSVC())) 
    CLASSIF =
        ("DecisionTree", DecisionTreeClassifier()) |
        ("RandomForest", RandomForestClassifier()) |
        ("Gradient Boosting Classifier", GradientBoostingClassifier()) |
        ("LogisticRegression", LogisticRegression()) |
        ("NearestNeighborClassifier", NearestNeighbors())
end


cfe = Herb.HerbSearch.ContextFreeEnumerator(grammar, 5, :START)
# println(length(Herb.HerbGrammar.nonterminals(grammar)))
# println(length(grammar.types))
rule1 = collect(cfe)[1000]
rule2 = collect(cfe)[2000]
# e = Herb.HerbGrammar.rulenode2expr(rule.children[2], grammar)
println(Herb.HerbGrammar.rulenode2expr(rule1, grammar))
println(Herb.HerbGrammar.rulenode2expr(rule2, grammar))
println()
cross_over(rule1, rule2)
println()
println(Herb.HerbGrammar.rulenode2expr(rule1, grammar))
println(Herb.HerbGrammar.rulenode2expr(rule2, grammar))


# children = get_children(rule)
# println(Herb.HerbGrammar.rulenode2expr(rule, grammar))
# println(length(children))
# println(children)
# for child in children
#     println(Herb.HerbGrammar.rulenode2expr(child, grammar))
# end
# for i in 1:length(grammar.types)
#     if(e == grammar.rules[i])
#         println()
#         println(grammar.types[i])
#         println(grammar.rules[i])
#     end
# end