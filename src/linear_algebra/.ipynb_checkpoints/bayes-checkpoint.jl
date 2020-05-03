module Bayes
export StochMapDom, OpenBayesNet, FreeBayesFunctions,
  Ob, Hom, dom, codom, compose, id, otimes, mzero, braid,
  mcopy, delete

import Base: +
using AutoHashEquals
using LinearMaps
import LinearMaps: adjoint
const LMs = LinearMaps
using LinearOperators
import LinearOperators:
  adjoint, opEye, opExtension, opRestriction, opZeros

using BayesNets
import BayesNets:
  BayesNet
  
using LightGraphs
import LightGraphs:
  indegree, outdegree, nv

using ...Catlab, ...Doctrines
import ...Doctrines:
  Ob, Hom, dom, codom, compose, ⋅, ∘, id, oplus, ⊕, mzero, braid,
  dagger, dunit, dcounit, mcopy, Δ, delete, ◊, mmerge, ∇, create, □,
  plus, zero, coplus, cozero, meet, top, join, bottom
using ...Programs
import ...Programs: evaluate_hom




@syntax FreeBayesFunctions(ObExpr,HomExpr) MonoidalCategoryWithDiagonals begin
  otimes(A::Ob, B::Ob) = associate_unit(new(A,B), mzero)
  otimes(f::Hom, g::Hom) = associate(new(f,g))
  compose(f::Hom, g::Hom) = new(f,g; strict=true) # No normalization!
end


# BayesMaps instance
#--------------------

@auto_hash_equals struct StochMapDom
  N::Int #Number of nodes in StochMapDom
end

struct OpenBayesNet
  Net::BayesNet
  dom::Array{Int}
  codom::Array{Int}
end
  
@instance MonoidalCategoryWithDiagonals(StochMapDom,OpenBayesNet) begin
#   @import adjoint, +

  dom(f::OpenBayesNet) = begin
#     f_dag = f.Net.dag
# #     vIDtoSymbol_dict = Dict()
# #     for entry in f.name_to_index;    
# #       vIDtoSymbol_dict[f.name_to_index[entry[1]]] = entry[1]
# #     end
#     indeg_bayes = indegree(f_dag)
#     names = []
#     for i in 1:size(indeg_bayes,1)
#         if indeg_bayes[i] == 0
#             push!(names,i)
#         end
#     end
    StochMapDom(length(f.dom))
  end
  codom(f::OpenBayesNet) = begin
#     f_dag = f.dag
#     indeg_bayes = indegree(f_dag)
#     names = []
#     for i in 1:size(indeg_bayes,1)
#         if indeg_bayes[i] != 0
#             push!(names,i)
#         end
#     end
    StochMapDom(length(f.codom))
  end

  compose(f::OpenBayesNet, g::OpenBayesNet) = begin
    dom = f.dom
    codom = [length(f.dom)+ elem for elem in g.codom]
    combined_net = deepcopy(f.Net)
    for elem in g.codom
      push!(combined_net, g.cpds[elem])
    end
    combined_dict = merge(f.Net.name_to_index, g.Net.name_to_index)
    for (key,value) in g.Net.name_to_index
      if value <= length(g.dom)
        !delete(combined_dict, key)
      else
        combined_dict[key] = value + length(f.dom)
      end
    end
    OpenBayesNet(BayesNet(combined_net.dag, combined_net.cpds, combined_dict), dom, codom)
  end
  
  #id(V::LinearMapDom) = LMs.UniformScalingMap(1, V.N)

  otimes(V::StochMapDom, W::StochMapDom) = begin
    StochMapDom(V.N + W.N)
  end
  
  otimes(f::OpenBayesNet, g::OpenBayesNet) = begin
    dom = vcat(f.dom, [nv(f.Net.dag) + elem for elem in g.dom])
    codom = vcat(f.codom, [nv(f.Net.dag) + elem for elem in g.codom])
    combined_dag = blockdiag(f.Net.dag, g.Net.dag)
    combined_cpd = vcat(f.Net.cpds, g.Net.cpds)
    combined_dict = merge(f.Net.name_to_index, g.Net.name_to_index)
    for (key,value) in g.Net.name_to_index
      combined_dict[key] = value + nv(f.Net.dag)
    end
    OpenBayesNet(BayesNet(combined_dag, combined_cpd, combined_dict), dom, codom)
  end
  
  id(V::StochMapDom) = begin
    OpenBayesNet(BayesNet(),[],[])
  end
  munit(::Type{StochMapDom}) = OpenBayesNet(BayesNet(),[],[])
#   mzero(::Type{LinearMapDom}) = LinearMapDom(0)
  braid(V::StochMapDom, W::StochMapDom) = begin
    OpenBayesNet(BayesNet(),[],[])
  end
  mcopy(V::StochMapDom) = OpenBayesNet(BayesNet(),[],[])
  delete(V::StochMapDom) = OpenBayesNet(BayesNet(),[],[])
#   plus(V::LinearMapDom) = LinearMap(plus_lm, mcopy_lm, V.N, 2*V.N)
#   zero(V::LinearMapDom) = LinearMap(zero_lm(V.N), delete_lm, V.N, 0)

#   plus(f::LinearMap, g::LinearMap) = f+g
#   scalar(V::LinearMapDom, c::Number) = LMs.UniformScalingMap(c, V.N)
#   antipode(V::LinearMapDom) = LMs.UniformScalingMap(-1, V.N)
end

# braid_lm(n::Int) = x::AbstractVector -> vcat(x[n+1:end], x[1:n])
# mcopy_lm(x::AbstractVector) = vcat(x, x)
# delete_lm(x::AbstractVector) = eltype(x)[]
# plus_lm(x::AbstractVector) = begin
#   n = length(x) ÷ 2
#   x[1:n] + x[n+1:end]
# end
# zero_lm(n::Int) = x::AbstractVector -> zeros(eltype(x), n)

end
