#Number recognizing network
#imports
using ScikitLearn
using CSV
using Statistics
using Random
using Plots
using Flux
using Flux: onehotbatch, onecold
using Flux.Tracker: update!, param, back!, grad

@sk_import datasets: load_digits
@sk_import model_selection: train_test_split

function gradient(X, y, W1, b1, W2, b2, W3, b3, a) #gradient function
    back!(cost_log(X, y, W1, b1, W2, b2, W3, b3))
    update!(W1, -a*grad(W1)); 
    update!(b1, -a*grad(b1));
    update!(W2, -a*grad(W2));
    update!(b2, -a*grad(b2));
    update!(W3, -a*grad(W3));
    update!(b3, -a*grad(b3));
    return  W1, b1, W2, b2, W3, b3
end

function GD(X, y, epochs, a) #gradient descent function
    W1 = param(rand(20, size(X,1)) .* 0.0001) #weights: multiply random numbers to make them smaller 
    b1 = param(rand(20)) #bias
    #second layer: 10 edges
    W2 = param(rand(15, 20) .* 0.0001) #weights 
    b2 = param(rand(15)) #bias
    #third layer: 1 edge
    W3 = param(rand(10, 15) .* 0.0001) #weights 
    b3 = param(rand(10)) #bias
    for i in 1:epochs
        W1, b1, W2, b2, W3, b3 = gradient(X, y, W1, b1, W2, b2, W3, b3, a)
    end
    #c = cost_log(X, y, W1, b1, W2, b2, W3, b3)
    return W1, b1, W2, b2, W3, b3
end

function cost_log(X, y, W1, b1, W2, b2, W3, b3) 
    ypred = modelpred(X, W1, b1, W2, b2, W3, b3)
    return mean(loss.(ypred, y))
end

#accuracy
accuracy(y, ypreds) = mean(Flux.onecold(ypreds) .== onecold(y))

#logistic prediction and cost functions
loss(ypred, y) = (-y * log(ypred)) - ((1-y)*log10(1-ypred))

#Modelpredictions
modelpred(X, W1, b1, W2, b2, W3, b3) = sigma.(layer(reluu.(layer(reluu.(layer(X, W1, b1)), W2, b2)), W3, b3))

#Reluu function
reluu(X) = max(0,X)
sigma(X) = max(min((1) / (1 + exp(-X)), 0.99999), 0.00001)

#Layer function
layer(X, W, b) = W * X .+ b

function main(X, y, Y, a)
    #Split, train and test
    X_train, X_test, y_train, y_test = train_test_split(X', y, train_size=0.81, test_size=0.19, random_state=3, stratify=y)
    X_train = Matrix(X_train)'
    X_test = Matrix(X_test)'
    ## One hot encode y
    Y_train =  Int32.(onehotbatch(y_train, 0:9))
    Y_test =  Int32.(onehotbatch(y_test, 0:9))
    #Train model
    W1, b1, W2, b2, W3, b3 = GD(X_train, Y_train, 2000, 0.5)
    #Make predictions
    ypred = modelpred(X_test, W1, b1, W2, b2, W3, b3)
    #get accuracy
    acc = accuracy(Y_test, ypred)
    return ypred, acc
end 

#Load the data
@sk_import datasets: load_digits
@sk_import model_selection: train_test_split
digits = load_digits();
X = Float32.(transpose(digits["data"]));  # make the X Float32 to save memory
y = digits["target"];
Y = Int32.(onehotbatch(y, 0:9));
a=0.01

#run the model
predictions, acc = main(X, y, Y, a) #accuracy: 0.9708









#-----------------------------------------------------------------------------------------------------------------------------------#
#code with readymade functions
using Flux
using Flux.Tracker: param, back!, grad
using Flux: onecold, crossentropy, throttle, onehotbatch
using ScikitLearn
using Base.Iterators: repeated
using StatsBase: sample

## Load the data
@sk_import datasets: load_digits
@sk_import model_selection: train_test_split
digits = load_digits();
X = Float32.(transpose(digits["data"]));  # make the X Float32 to save memory
y = Int32.(digits["target"]);
Y = onehotbatch(y, 0:9);
nfeatures, nsamples = size(X)

## Split train and test
X_train, X_test, y_train, y_test = train_test_split(X', y, train_size=0.81, test_size=0.19, random_state=3, stratify=y)
X_train = X_train'
X_test = X_test'

## One hot encode y
Y_train = onehotbatch(y_train, 0:9)
Y_test = onehotbatch(y_test, 0:9)

## Build a network. The output layer uses softmax activation, which suits multiclass classification problems.
model = Chain(
  Dense(nfeatures, 32, Flux.relu),
  Dense(32, 10),
  softmax
)

## cross entropy cost function
cost(x, y) = crossentropy(model(x), y)

opt = Descent(0.005)  # Choose gradient descent optimizer with alpha=0.005
dataset = repeated((X_train, Y_train), 2000)  # repeat the dataset 2000 times, equivalent to running 2000 iterations of gradient descent
Flux.train!(cost, params(model), dataset, opt)

accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
accuracy(X_test, Y_test)  # 0.953216