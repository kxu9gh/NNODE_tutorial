# Explore different examples which use a neural net in the rhs of an ODE system
# Ke Xu, Jun Allard allardlab.com

# set up environment (uses contents of Project.toml and Manifest.toml)
using Pkg; Pkg.activate("."); Pkg.instantiate()

# importing packages
using Lux, DiffEqFlux, Optimization, OptimizationOptimJL, DifferentialEquations, Random, Plots

# Create a name for saving ( basically a prefix )
svname = "Scenario_3_"

rng = Random.default_rng()
Random.seed!(rng,1)

#### ------ EXAMPLE 3: 3d ODE system, independent.   ---------
# One variable has rhs with NN term and vector parameter term
# One variable has rhs with NN
# One variable has rhs with vector parameter

## ---------------------------------------------------------------------------
## ------ 1. Ground Truth ------
println("1. Define the ground truth model")
## ---------------------------------------------------------------------------

function term_to_be_learned(u,p,t)
    (p[1] - u[1].^1.8) / p[2]
end

function dudt_groundtruth(du,u,p,t)
  du[1] = term_to_be_learned(u,p,t) - p[3].*u[1]
  du[2] = -p[3]*u[2]
  du[3] = -p[4]*u[3]
end

p_groundtruth = Float32[0.2,0.8,1,1.6]  # ground truth parameters
u0 = [1f0;1f0;1f0] # initial condition
tspan = (0f0,10f0)


## ---------------------------------------------------------------------------
## ------ 2. Generate synthetic data ------
println("2. Generate synthetic data")
## ---------------------------------------------------------------------------

prob = ODEProblem(dudt_groundtruth,u0,tspan,p_groundtruth)
sol_groundtruth = solve(prob, Tsit5(), saveat = 0.5 )

X = Array(sol_groundtruth)
XSynthetic = X + Float32(1e-3)*randn(eltype(X), size(X))  #noisy data

timepoints = sol_groundtruth.t


## ---------------------------------------------------------------------------
## ------ 3. Model definition ------
println("3. Define the model to use for learning")
## ---------------------------------------------------------------------------

# Neural Network for first term
NN_1 = Lux.Chain(Lux.Dense(1, 16, tanh), Lux.Dense(16, 1))
p1,structure1 = Lux.setup(rng, NN_1)

# Neural Network for second term
NN_2 = Lux.Chain(Lux.Dense(1, 16, tanh), Lux.Dense(16, 1))
p2, structure2 = Lux.setup(rng, NN_2)

# Vector of parameters
p_initialGuess = [20.0f0]

p3 = Lux.ComponentArray{eltype(p_initialGuess)}()
p3 = Lux.ComponentArray(p3;p_initialGuess)

# combine lux models

p1 = Lux.ComponentArray(p1)
p2 = Lux.ComponentArray(p2)

# combining parameters into one
p = Lux.ComponentArray{eltype(p1)}()
p = Lux.ComponentArray(p;p1)
p = Lux.ComponentArray(p;p2)
p = Lux.ComponentArray(p;p3)


# NN([x], p, st)[1][1]: outputs, 1st dynamic variable
# NN([x], p, st)[1][2]: outputs, 2nd dynamic variable
# NN([x], p, st)[2]: structure of the neural network 
function dudt_model(u,p,t)
    v,w = u
    du1 = NN_1([v], p.p1, structure1)[1][1] - p_groundtruth[3]*u[1] 
    du2 = NN_2([w], p.p2, structure2)[1][1]
    du3 = -p.p3[1]*u[3]
    [du1,du2,du3] # this function needs to return an array for ODEProblem
end
prob_nn = ODEProblem(dudt_model,u0, tspan, p)
sol_nn = solve(prob_nn, Tsit5(),saveat = timepoints) # generate a solution at the initial guess (I think)

## ---------------------------------------------------------------------------
## ------ 4. Set up learning procedure ---------------------------------------
println("4. Set up learning procedure")
## ---------------------------------------------------------------------------


function predict(theta)
    Array(solve(prob_nn, Vern7(), p=theta, saveat = timepoints,
                         abstol=1e-6, reltol=1e-6,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end

# No regularisation right now
function loss(theta)
    pred = predict(theta)
    sum(abs2, XSynthetic .- pred), pred # lsq
end

loss(p) 
losses = []
allplots = []

callback(theta,l,pred; doplot=true) = begin
    push!(losses, l)
    if length(losses)%50==0
        println(losses[end])
    end
    if doplot
        plt = scatter(sol_groundtruth, label = ["data 1" "data 2" "data 3"], color=[:orange :green :blue], legend=:bottomleft)
        plot!(plt, timepoints, transpose(pred), label = ["pred 1 (combo)" "pred 2 (NN)" "pred 3 (vector"], color=[:orange :green :blue])
        push!(allplots, transpose(pred))
        # display(plt)
    end
    false
end

#callback(p, loss(p)...)

## ---------------------------------------------------------------------------
## ------ 5. LEARN! ------
println("5. Learn!")
## ---------------------------------------------------------------------------

adtype = Optimization.AutoZygote()
optimization_function = Optimization.OptimizationFunction((x,p) -> loss(x), adtype)

optimization_problem_step1 = Optimization.OptimizationProblem(optimization_function, p)
learned_nnode_step1 = Optimization.solve(optimization_problem_step1, ADAM(0.01), callback=callback, maxiters = 1000) # currently BFGS is sensitive to ADAM convergence

println("Finished ADAM, beginning BFGS")

optimization_problem_step2 = Optimization.OptimizationProblem(optimization_function, learned_nnode_step1.u)
learned_nnode_step2 = Optimization.solve(optimization_problem_step2, BFGS(), maxiters = 10000, callback = callback)

learned_nnode = learned_nnode_step2

## ---------------------------------------------------------------------------
## ------ 6. Show model prediction and learned model ------
println("6. Show model prediction and learned model innards")
## ---------------------------------------------------------------------------

## 6.1 Show the final predicted time series

plt = scatter(sol_groundtruth, label = ["data 1" "data 2" "data 3"], color=[:orange :green :blue], legend=:topright, ylims = [0.0, 1.2])
plot!(plt, timepoints, allplots[end], label = ["pred 1 (combo)" "pred 2 (NN)" "pred 3 (vector)"], color=[:orange :green :blue])
savefig(joinpath(pwd(), "plots", "$(svname)01TimeSeriesTruePred.pdf"))

## 6.2 Show the model internals

y_axis = range(0.0, 1.0, 100) #transpose(loss(res2_uode.u)[2])

function NN_func(y) 
    NN_1([y], learned_nnode.u.p1, structure1)[1][1]
end

function term_to_be_learned_func(y)
    term_to_be_learned(y,p_groundtruth,0)
end

plot(y_axis, [NN_func.(y_axis), term_to_be_learned_func.(y_axis)], label = ["Neural Net prediction" "true interaction"])
savefig(joinpath(pwd(), "plots", "$(svname)02NNFunctionTruePred.pdf"))

## 6.3 Make an animation of learning process, of the time series
anim = @animate for i in eachindex(allplots)
    plt = scatter(sol_groundtruth, label = ["data 1" "data 2" "data 3"], color=[:orange :green :blue], legend=:bottomleft, ylims = [-3.0, 2.0])
    plot!(plt, timepoints, allplots[i], label = ["pred 1 (combo)" "pred 2 (NN)" "pred 3 (vector"], color=[:orange :green :blue])
end

gif(anim, joinpath(pwd(), "plots", "$(svname)03LearningPrediction.mp4"), fps=60)

## 6.4 Print to screen the vector parameters, to compare ground truth and learned value
println("p[4] ground truth=$(p_groundtruth[4]), p[4] learned=$(learned_nnode.u.p3[1])")