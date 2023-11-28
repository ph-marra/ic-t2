module Test

using Flux, MLDatasets, CUDA, Statistics
include("configs.jl")
include("dataset.jl")

x_test, y_test = Dataset.get_dataset("test") # |> gpu

mean_x_test = mean(x_test)
std_x_test = std(x_test)

# normalization
x_test = (x_test .- mean_x_test) ./ std_x_test

y_test = Flux.onehotbatch(y_test, Configs.SPLIT)

end