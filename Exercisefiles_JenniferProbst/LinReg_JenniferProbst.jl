import Plots
using Plots
using VegaDatasets
using DataFrames
using LinearAlgebra

function plotting(x, y, a, b, title="Plot", xlab="xlab", ylab="ylab")
    equation(d)=a*d+b
    plot(x, y, seriestype=:scatter, title=title, xlabel=xlab, ylabel=ylab)
    plot!(equation, minimum(x):maximum(x))
end

##Exercise 
#Linear Regression
using RDatasets
trees = DataFrame(RDatasets.dataset("datasets", "trees"))
x = trees[!, :Girth]
y = trees[!, :Height]
    
#analytical solution
include("Lossfunction_JenniferProbst.jl") #get the optimize function
a = collect(range(-2, length=1000, stop=2))
b = collect(range(60, length=1000, stop=80))
bestssq1, besta1, bestb1, bestssq2, besta2, bestb2 = optimize(x, y, a, b)
plotting(x, y, besta1, bestb1)
plotting(x, y, besta2, bestb2)

#gradient descent (GD) --> need row of 1nes in X data
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

x= trees[!, :Girth]
r = ones(length(x))
X=hcat(r,x)
Y = trees[!, :Height]
values, cos = GD_lin(X, y, 5000000, 0.0003)
plotting(x,y,values[2], values[1], "Plot of Girth and Height", "Girth of Trees", "Heigth of Trees")