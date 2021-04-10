import Plots
using Plots
using VegaDatasets
using DataFrames
using LinearAlgebra
using Statistics

#Plotting 2D
function plotting_2D(x, y, a, b, title="Plot", xlab="xlab", ylab="ylab")
    equation(d)=a*d+b
    plot(x, y, seriestype=:scatter, title=title, xlabel=xlab, ylabel=ylab)
    plot!(equation, minimum(x):maximum(x))
end

#Linear Regression functions
function cost_lin(X, Y, A)
    return 1/(2*size(X, 1)) * sum((X*A-Y).^2)
end

function gradient_lin(X, Y, A, alp)
    deriv=1/size(X,1) .* (X*A .- Y)' * X
    newA= A' .- (alp.*deriv)
    return newA
end

function GD_lin(X, Y, epochs, alp) 
    A=Array{Int64}(undef, size(X)[2])
    for i in 1:epochs
        newA = gradient_lin(X, Y, A, alp)'
        A=newA
    end
    current_cost=cost_lin(X, Y, A)
    return A, current_cost
end

#Ridge Regression function
function cost_rid(X, Y, A, lambda, n)
    return 1/2*n * (sum((X*A .- Y).^2)+lambda*sum(A[2:end].*A[2:end]))
end

function gradient_rid(X, Y, A, alp, lambda, n)
    newA=deepcopy(A)
    newA[1]=0
    deriv=1/n .* ((X*A .- Y)'*X)'.+(lambda*newA/n)
    new= A .- (alp.*deriv)
    return new
end

function GD_rid(X, Y, epochs, alp, lambda, n) 
    A=rand(size(X)[2])
    for i in 1:epochs
        newA = gradient_rid(X, Y, A, alp, lambda, n)
        A=newA
    end
    current_cost=cost_rid(X, Y, A, lambda, n)
    return A, current_cost
end

#Minmaxnorm function
function minmaxnorm(column)
    normcolumn=rand(length(column))
    for i in 1:length(column)
        normcolumn[i]=(column[i]-minimum(column))/(maximum(column)-minimum(column))
    end
    return normcolumn
end

##Exercise: tree dataset 2D --> linear and ridge comparison
import VegaDatasets
import DataFrames
using RDatasets
trees = DataFrame(RDatasets.dataset("datasets", "trees"))
x= trees[!, :Girth]
r = ones(length(x))
X=Matrix(hcat(r,x))
Y = trees[!, :Volume]
n=size(X, 1)
lambda=0.05
values_lin, cos_lin = GD_lin(X, Y, 5000000, 0.0003) #cost: 8.4564, values: -36.94345912434657; 5.06585642282715
values_rid, cos_rid = GD_rid(X, Y, 5000000, 0.0003, lambda, n) #cost: 8146.5647 (much higher because rescaled), values: -36.93210256185521, 5.064999219425076 (same values)
plotting_2D(x, Y, values_lin[2], values_lin[1], "Linear prediction of tree height", "Girth", "Tree Volume")
plotting_2D(x, Y, values_rid[2], values_rid[1], "Ridge-prediction of tree height", "Girth", "Tree Volume")
#predict volumes: almost the sames
volumes_lin = X*values_lin
volumes_rid = X*values_rid


#Exercise: tree dataset 3D
trees = DataFrame(RDatasets.dataset("datasets", "trees"))
x= trees[!, [:Girth, :Height]]
n=size(X, 1)
r = ones(n)
X=Matrix(hcat(r,x))
Y = trees[!, :Volume]
lambda=0.05
values_lin, cos_lin = GD_lin(X, Y, 5000000, 0.0003) #cost: 8.4564, values: 46722.25884300665; 275.9945085579061; -658.4724160348123
values_rid, cos_rid = GD_rid(X, Y, 5000000, 0.0003, lambda, n) #cost: 8146.5647, values: -57.98989590899273, 4.707109169932937, 0.3394642229579661
#predicted volume
volumes_lin2= X*values_lin #not exactly the same as volmes_lin
volumes_rid2=X*values_rid

#Separate a random 20% of the data for testing. Train your model with the remaining 80% of the data and test its performance on the test data.
#select 80%, let model run, predict values for other 20, calculate cost for them
using Random
ran=shuffle(Vector(1:n))
choose=ran[Int(ceil(0.2*n)):end]
trainX=Matrix(X[choose,:])
newn=size(trainX, 1)
values_80r, cos_80r = GD_rid(X, Y, 5000000, 0.00003, lambda, newn) #cost: 6569.8185, estimates:-36.92934171367576, 5.064801507463253
#test data
choosenot=ran[1:Int(floor(0.2*n))]
predictX=Matrix(X[choosenot,:])
predictY=Y[choosenot, :]
cost_rid(predictX, predictY, values_80r, lambda, newn)


#Test higher degree polynomials
trees = DataFrame(RDatasets.dataset("datasets", "trees"))
x= trees[!, [:Girth, :Height]]
x.Girth=minmaxnorm(x.Girth)
x.Girth2=minmaxnorm(x.Girth.^2)
x.Height=minmaxnorm(x.Height)
x.Heigth2=minmaxnorm(x.Height.^2)
n=size(X, 1)
r = ones(n)
X=Matrix(hcat(r,x))
Y = minmaxnorm(trees[!, :Volume])
lambda=0.05
values_lin, cos_lin = GD_lin(X, Y, 5000000, 0.0003) #cost: 0.000672
values_rid, cos_rid = GD_rid(X, Y, 5000000, 0.0003, lambda, n) #cost: 0.979329


#Compare your results with the results of ScikitLearn’s built-in functions
#ScikitLearn functions
using RDatasets
using Plots
using ScikitLearn
import ScikitLearn: fit!, predict
@sk_import linear_model: Ridge
@sk_import linear_model: Lasso

function Ridgef(x,y)
    model_ridge= Ridge(alpha=1.0)
    fit!(model_ridge, x, y)
    newY = predict(model_ridge, x)
    costlin=1/2*n * (sum((newY .- Y).^2)) 
    return costlin
end

function Lassof(x, y)
    model_lasso= Lasso(alpha=1.0)
    fit!(model_lasso, x, y)
    newY = predict(model_lasso, x)
    costlin=1/2*n * (sum((newY .- Y).^2))
    return costlin
end

trees = DataFrame(RDatasets.dataset("datasets", "trees"))
y=trees[!, :Height]
x=Matrix(trees[!, [:Girth, :Height]])
Ridgef(x, y) #linear cost: 2.771589 compared to 8.4564 with my data
Lassof(x, y) #linear cost: 2.771589