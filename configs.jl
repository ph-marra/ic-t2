module Configs

using Flux, MLDatasets, Statistics

N=64
PI=2
PJ=2
DR=0.5
LD=10
A=Flux.relu
F=Flux.softmax

conv_type1(in, out) = Flux.Conv((3, 3), in => out, relu, pad=(1, 1), stride=(1, 1))
pool(i,j) = Flux.MaxPool((i,j))

SPLIT=0:9
BATCHSIZE=128
OPT = Flux.ADAM(0.001)
N_EPOCHS = 10

function accuracy(y_, y)
    return Statistics.mean(Flux.onecold(y_) .== Flux.onecold(y))
end

end


