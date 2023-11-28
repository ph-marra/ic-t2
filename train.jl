module Train

using Flux, MLDatasets, CUDA, Statistics
include("configs.jl")
include("dataset.jl")

class_names, x_train, y_train = Dataset.get_dataset("train")# |> gpu

mean_x_train = mean(x_train)
std_x_train = std(x_train)

# normalization
x_train = (x_train .- mean_x_train) ./ std_x_train

y_train = Flux.onehotbatch(y_train, Configs.SPLIT)
train_data = Flux.Data.DataLoader((x_train, y_train), batchsize=Configs.BATCHSIZE)

end