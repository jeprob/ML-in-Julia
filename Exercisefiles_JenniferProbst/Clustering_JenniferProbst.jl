#K-means clustering
using Plots
using ScikitLearn
using Random
using Statistics
@sk_import datasets: samples_generator

function randominitializer(lengthx, k) #gives indices of startcentroids
    return shuffle(Vector(1:length(lengthx)))[1:k]
end

function distance(point, centroid) #gives distance from one point to given centroid
    return 1/(length(point)) * sum((point .- centroid).^2)
end

function newcentroids(data, k, groups, centroids) #calculate new centroids based on means of the groups
    for j in 1:k
        centroids[j,1] = Statistics.mean(data[findall(x -> x == j, groups),1])
        centroids[j,2] = Statistics.mean(data[findall(x -> x == j, groups),2])
    end
    return centroids
end

function main(data, k, param)
    startind = randominitializer(data[:, 1], k) #choose random indices
    centroids = data[startind, :] #and centroids
    groups=[]
    for m in 1:param #number of iterations
        groups=[]
        for i in 1:length(data[:, 1])
            distances=[]
            for j in 1:size(centroids,1)
                push!(distances, distance(data[i, :], centroids[j, :]))
            end
            a = findmin(distances)[2] #assign to group where it has minimum distance
            push!(groups, a)
        end
        centroids=newcentroids(data, k, groups, centroids) #update centroids based on assignment
    end
    return centroids, groups
end 

#run
k=5 #number of clusters
features, labels = samples_generator.make_blobs(n_samples=500, centers=k, cluster_std=0.55, random_state=0) #generate samples
param=100 #number of iterations
centroids, endassignment = main(features, k, param) 
scatter(features[:, 1], features[:, 2], color=endassignment, title = "K-means clustering output")

#same with ScikitLearn
import ScikitLearn: fit!, predict
@sk_import cluster: KMeans

model = KMeans(n_clusters=k, tol=0.0001, max_iter=10_000, init="random")

fit!(model, features)
assignment = predict(model, features) .+ 1
scatter(features[:, 1], features[:, 2], color=assignment, title = "K-means clustering output")



#----------------------------------------------------------------------------------------------------------------------------------#
#Decision tree with ScikitLearn
@sk_import tree: DecisionTreeClassifier

# Load the data
@sk_import datasets: load_breast_cancer
all_data = load_breast_cancer()
X = all_data["data"];
y = all_data["target"];

# Create a tree
model = DecisionTreeClassifier(min_samples_split=2, random_state=0, min_samples_leaf=1, max_depth=4)
fit!(model, X, y)

model.feature_importances_

# Plot the tree
using PyCall
@sk_import tree: export_graphviz
export_graphviz(model, out_file="mytree", class_names=["Healthy", "Cancerous"], feature_names=all_data["feature_names"], leaves_parallel=true, impurity=false, rounded=true, filled=true, label="root", proportion=true)

# you need to install graphviz for python first. --> does not work at all
dot_graph = read("mytree", String)
j = graphviz.Source(dot_graph, "mytree", "./", "pdf", "dot")
j.render()