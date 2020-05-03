module TestBayes
using Test

using Random
using IterativeSolvers

using Catlab, Catlab.Doctrines, Catlab.WiringDiagrams, Catlab.Programs
using Catlab.LinearAlgebra.Bayes

import LinearAlgebra: norm, svd

import LinearMaps: BlockDiagonalMap, UniformScaling
using LinearOperators

using BayesNets
import BayesNets:
  BayesNet
# Evaluation
############

# BayesMap instance
#--------------------

#Create Equations
a_equation = randn(100)
b_equation = randn(100) .+ 2*a_equation .+ 3
c_equation = randn(100) .+ 2*b_equation .+ 3

#Build Rain, Sprinkler, GrassWet Example BayesNet
data = DataFrame(rain=a_equation, sprinkler=b_equation, grasswet=c_equation)
cpdA = fit(StaticCPD{Normal}, data, :rain)
cpdB = fit(LinearGaussianCPD, data, :sprinkler, [:rain])
cpdC = fit(LinearGaussianCPD, data, :grasswet, [:rain,:sprinkler])

bn1 = BayesNet([cpdA, cpdB, cpdC])
dom = [1]
codom = [2,3]

openbn1 = OpenBayesNet(bn1, dom, codom)


#Create Equations
a_equation = randn(100)
b_equation = randn(100) .+ 2*a_equation .+ 3

#Build Rain, Sprinkler, GrassWet Example BayesNet
data = DataFrame(mosquito=a_equation, bird=b_equation)
cpdA = fit(StaticCPD{Normal}, data, :mosquito)
cpdB = fit(LinearGaussianCPD, data, :bird, [:mosquito])

bn2 = BayesNet([cpdA, cpdB])
dom = [1]
codom = [2]

openbn2 = OpenBayesNet(bn2, dom, codom)

openbn_otimes = otimes(openbn1, openbn2)
@test dom(openbn_otimes) = dom(openbn1) + dom(openbn2)
@test codom(openbn_otimes) = codom(openbn1) + codom(openbn2)

# @test (f⋅h)*x == h*(f*x)
# @test (f⊕g)*[x;y] == [f*x; g*y]
# @test braid(dom(f),dom(g)) * [x;y] == [y;x]

# @test mcopy(dom(f))*x == [x;x]
# @test delete(dom(f))*x == []
# @test plus(dom(f))*[x;x] == 2x
# @test zero(dom(f))*Float64[] == zero(x)

# @test (h+k)*x == h*x + k*x
# @test scalar(dom(f),3)*x == 3*x
# @test antipode(dom(f))*x == -1*x
# @test adjoint(g)*y == g'*y
end