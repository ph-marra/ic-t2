using Flux, CUDA, DataFrames, CSV, BSON

include("configs.jl")
include("train.jl")
include("test.jl")

nn_model = Chain(
    Configs.conv_type1(3, Configs.N), BatchNorm(Configs.N),   

    Configs.conv_type1(Configs.N, Configs.N), BatchNorm(Configs.N), Configs.pool(Configs.PI, Configs.PJ),

    Configs.conv_type1(Configs.N, 2*Configs.N), BatchNorm(2 * Configs.N),
    Configs.conv_type1(2*Configs.N, 2*Configs.N), BatchNorm(2 * Configs.N), Configs.pool(Configs.PI, Configs.PJ),

    Configs.conv_type1(2*Configs.N, 4*Configs.N), BatchNorm(4 * Configs.N),
    Configs.conv_type1(4*Configs.N, 4*Configs.N), BatchNorm(4 * Configs.N),
    Configs.conv_type1(4*Configs.N, 4*Configs.N), BatchNorm(4 * Configs.N),
    Configs.conv_type1(4*Configs.N, 4*Configs.N), Configs.pool(Configs.PI, Configs.PJ),

    Configs.conv_type1(4*Configs.N, 8*Configs.N), BatchNorm(8 * Configs.N),
    Configs.conv_type1(8*Configs.N, 8*Configs.N), BatchNorm(8 * Configs.N),
    Configs.conv_type1(8*Configs.N, 8*Configs.N), BatchNorm(8 * Configs.N),
    Configs.conv_type1(8*Configs.N, 8*Configs.N), Configs.pool(Configs.PI, Configs.PJ),
    
    Configs.conv_type1(8*Configs.N, 8*Configs.N), BatchNorm(8 * Configs.N),
    Configs.conv_type1(8*Configs.N, 8*Configs.N), BatchNorm(8 * Configs.N),
    Configs.conv_type1(8*Configs.N, 8*Configs.N), BatchNorm(8 * Configs.N),
    Configs.conv_type1(8*Configs.N, 8*Configs.N), Configs.pool(Configs.PI, Configs.PJ),

    x -> reshape(x, :, size(x, 4)),
    Dense(8*Configs.N, 64*Configs.N, Configs.A),
    Dropout(Configs.DR),
    Dense(64*Configs.N, 64*Configs.N, Configs.A),
    Dropout(Configs.DR),
    Dense(64*Configs.N, Configs.LD),
    Configs.F
)# |> gpu

loss(x, y) = Flux.crossentropy(nn_model(x), y)

function run_model(th=1e-03, ep=5, eta=1e-06, conv=10)
    best_acc = 0
    last_improv = 0
    ps = Flux.params(nn_model)
    
    dfm = DataFrame([[], [], []], ["Epoch (N)", "Accuracy (%)", "Time (s)"])

    for e in range(1, Configs.N_EPOCHS)
        t = @time Flux.train!(loss, ps, Train.train_data, Configs.OPT)

        ŷ_test = nn_model(Test.x_test)
        acc = Configs.accuracy(ŷ_test, Test.y_test)

        push!(dfm, [e, acc, t])

        BSON.@save "trained_model_" * String(Symbol(e)) * ".bson" nn_model
        
        print("\n--------------------------------\n")
        print(nn_model)
        print("\n--------------------------------\n")
        print(dfm)
        print("\n--------------------------------\n")

        if acc >= best_acc
            best_acc = acc
            BSON.@save "trained_model_" * "(best=" * String(Symbol(e)) * ").bson" nn_model
            
            #print(nn_model)

            #let model = cpu(model)
            #  BSON.@save "./trained_model.bson" model
            #end

        end

        if acc >= 1 - th
            break
        end

        if e - last_improv >= ep && Configs.OPT.eta > eta
            Configs.OPT.eta /= 10.0

            last_improv = e
        end

        if e - last_improv >= conv
            break
        end

    end

    CSV.write("results.csv", dfm)
end

run_model()
