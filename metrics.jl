using LinearAlgebra, Statistics, Flux, MLDatasets, CUDA
using BetaML: ConfusionMatrix, fit!, info
using Printf, BSON

include("dataset.jl")
include("test.jl")

best_model_file="trained_model_8.bson"

BSON.@load best_model_file nn_model

print("\n-----------------------------------------------\n")
print(nn_model)
print("\n-----------------------------------------------\n")

ŷ_test = nn_model(Test.x_test)

cm = ConfusionMatrix()
fit!(cm, Flux.onecold(Test.y_test) .- 1, Flux.onecold(ŷ_test) .- 1)

print(cm)
print("\n-----------------------------------------------\n")
print(info(cm))
print("\n-----------------------------------------------\n")