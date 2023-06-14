include("src/genetic_algorithm.jl")
include("../lib/herb.jl/src/herb.jl")
include("src/grammar.jl")

println("start")

# cfe = Herb.HerbSearch.ContextFreeEnumerator(grammar, 5, :START)
# rule = collect(cfe)[1000]

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
using Plots

# Sample data
# for i in (1:10)
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
results = Base.iterate(it)
println("PIPELINES: $(Herb.HerbGrammar.rulenode2expr(results.best_programs[length(results.best_programs)], grammar))")
    # fitnesses = results.best_program_fitness

    # for i in (1: length(fitnesses))
    #     fitnesses_avg[i] += fitnesses[i]/10
# end

# fitnesses = Base.iterate(it).best_program_fitness

plot(results.best_program_fitnesses, marker=:o, xlabel="iteration", ylabel="fitness", title="", ylim=(0, 1), xticks=1:length(results.best_program_fitnesses))

savefig("plot.png")
