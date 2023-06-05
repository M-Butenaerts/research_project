include("src/genetic_algorithm.jl")
include("../lib/herb.jl/src/herb.jl")
include("src/grammar.jl")

println("start")

cfe = Herb.HerbSearch.ContextFreeEnumerator(grammar, 5, :START)
rule = collect(cfe)[1000]

# Any[:Pipeline, :([("seq", Pipeline([("StandardScaler", StandardScaler()), ("par", FeatureUnion([("MinMaxScaler", MinMaxScaler()), ("PolynomialFeatures", PolynomialFeatures())]))])), ("NearestNeighborClassifier", NearestNeighbors())])]
# Pipeline([
#   ("seq", Pipeline([
#       ("StandardScaler", StandardScaler()),
#       ("par", FeatureUnion([
#           ("MinMaxScaler", MinMaxScaler()),
#           ("PolynomialFeatures", PolynomialFeatures())
#       ])
#   ]),
#   ("NearestNeighborClassifier", NearestNeighbors())                   
#])
# println(parser(Herb.HerbSearch.rulenode2expr(rule, grammar)))

it = Herb.HerbSearch.GeneticSearchIterator(
    grammar, 
    Vector{Herb.HerbData.IOExample}([]), 
    fitness_function_function(61),
    cross_over,
    mutation, 
    stopping_condition, 
    :START, 
    5, 
    10, 
    0.1, 
    0.1
)

Base.iterate(it)