#Exercise
#Reduce data to 2D with 2 random eigenvectors

using ScikitLearn
using Statistics
using LinearAlgebra
using Plots
using Random
@sk_import datasets: load_breast_cancer
@sk_import covariance: EllipticEnvelope
@sk_import datasets: make_moons
@sk_import datasets: make_blobs

function meannorm(data) #normalizing/scaling function
    return (data .- mean(data, dims=1)) ./ (maximum(data,dims=1) - minimum(data,dims=1))
end

function reductiontokD(X, y, k) #reduce data to k random dimensions
    X = meannorm(X)
    cv = Statistics.cov(X)
    ews, evs = LinearAlgebra.eigen(cv)
    newvec = evs[:, 1:k]
    return X*newvec
end

all_data = load_breast_cancer()
X = all_data["data"]
y = all_data["target"]
newx = reductiontokD(X, y, 2)
Plots.scatter(newx[:, 1], newx[:, 2])


#Reduce till 99% of var explained
function eigvar(eigenvalues) #calculates var explained by each eigenvalue
    eigenvalues ./ sum(eigenvalues)
end

function reduce99(X, y, cutoff)  #reduce data to k random dimensions till variance is explained
    X = meannorm(X)
    cv = Statistics.cov(X)
    ews, evs = LinearAlgebra.eigen(cv)
    varex = eigvar(ews)
    sum = 0 
    i=0
    newvec=[]
    while sum < cutoff
        i+=1
        newvec = evs[:, end-(i-1):end]
        sum+=varex[end-(i-1)]
    end
    return X*newvec
end 

newdata2 = reduce99(X, y, 0.99)
Plots.scatter(newdata2[:, 1], newdata2[:, 2])


#-----------------------------------------------------------------------------------------------------------------------------#
#outlier detection
function splitt(X, perc) #split in train and test dataset
    ran=shuffle(Vector(1:length(y)))
    choose=ran[Int(round(perc*length(y))):end]
    choosenot=ran[1:Int(floor(0.2*length(y)))]
    X_train=Matrix(X[choose,:])
    X_test=Matrix(X[choosenot,:])
    return X_train, X_test
end

function normaldensity(x, varianz, mu) #gives probability of one value for one dimension (normal distribution assumed)
    return (1/sqrt(2*pi*varianz))*exp(-(x-mu)^2/(2*varianz))
end

function calculateprob(newpoint, Xdata) #calculates the probability of one data point in multiple dimensions
    prob=1
    for i in 1:size(newpoint, 1)
        prob*=normaldensity(newpoint[i], Statistics.var(Xdata[:,i]), mean(Xdata[:,i]))
    end
    return prob
end

function outlierlist(X_train, X_test) #checks of probability of individual values with assumed underlaying normal distribution is <0.05
    ol=[]
    for i in 1:size(X_test, 1)
        pro=calculateprob(X_test[i, :], X_test)
        if pro < 0.05
            push!(ol, X_test[i, :])
        end
    end
    return ol
end

#example setting
n_samples = 300
outliers_fraction = 0.05
n_outliers = round(Int64, outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers
X, y = make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5, random_state=1, n_samples=n_inliers, n_features=2)
#run 
X_train, X_test = splitt(X, 0.1)
oll = outlierlist(X_train, X_test) #returns list of outlier elements
Plots.scatter(X_train[:, 1], X_train[:, 2])
Plots.scatter!(oll[1], oll[2], color="red")