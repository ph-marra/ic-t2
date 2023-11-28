module Dataset

using MLDatasets, CUDA
include("configs.jl")

function get_dataset(s)
    if s == "train"
        class_names = MLDatasets.CIFAR10().metadata["class_names"]
        x_train, y_train = MLDatasets.CIFAR10(Float32, split=:train)[:]
        return class_names, x_train, y_train
    else
        x_test, y_test = MLDatasets.CIFAR10(Float32, split=:test)[:]
        return x_test, y_test
    end
end

end

