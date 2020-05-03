module LinearAlgebra

using Reexport

include("GLA.jl")
include("StructuredGLA.jl")
include("bayes.jl")

@reexport using .GraphicalLinearAlgebra
@reexport using .StructuredGraphicalLinearAlgebra
@reexport using .Bayes
end
