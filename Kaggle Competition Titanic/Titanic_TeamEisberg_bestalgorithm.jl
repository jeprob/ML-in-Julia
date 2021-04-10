#onehot encoded the data, filled missing values and first try of different algorithms on the data
#trying just random parameter combinations we got the highest score achived with the decision tree classifier
#in the two other files submitted we got lower results (trying gridsearch, other parameters)

#imports
using DataFrames
using CSV
using Statistics
using LinearAlgebra

#minmax normalisation
function minmaxnorm(column)
  normcolumn=rand(length(column))
  for i in 1:length(column)
      normcolumn[i]=(column[i]-minimum(column))/(maximum(column)-minimum(column))
  end
  return normcolumn
end

# onehotencoder
function onehotencoder!(df::DataFrame, colname::Symbol)
  uniq = levels(df[!, colname])
  if findfirst(x-> x==true, ismissing.(df[!, colname])) != Nothing  # if there are missing values
    uniq = vcat(uniq, missing)
  end
  dfsize = size(df, 1)
  for value in uniq
    if !ismissing(value)
      null = zeros(Int64, dfsize)
      notnull = findall(x-> !ismissing(x) && x== value, df[!, colname])
      null[notnull] .= 1
      df[!, Symbol(value)] = null
    else
      null = zeros(Int64, dfsize)
      notnull = findall(x-> ismissing(x) , df[!, colname])
      null[notnull] .= 1
      df[!, Symbol(string(colname) * "_missing")] = null
    end
  end
  # Create a vector of column names excluding the `colname`
  newnames = names(df)
  splice!(newnames, findfirst(x-> x == colname, newnames))
  df = df[!, newnames]
  return df
end

#onehot encode the parameters
function onehot_titanic(dataframe)
    #embarked
    df_onehot = onehotencoder!(dataframe, :Embarked)    
    names!(df_onehot, [:PassengerId, :Pclass, :Name, :Sex, :Age, :SibSp, :Parch, :Ticket, :Fare, :Cabin, :Embarked_C, :Embarked_Q, :Embarked_S, :Embarked_missing])
    #passengerclass
    df_onehot = onehotencoder!(df_onehot, :Pclass)
    #cabin deck
    df_copy = deepcopy(df_onehot)
    for i in 1:size(df_onehot, 1)
      if !ismissing(df_onehot[i, :Cabin])
        c = df_onehot[i, :Cabin]
        df_copy[i, :Cabin] = string(c[1])
      end
    end
    df_onehot = onehotencoder!(df_copy, :Cabin)
    #Titles
    n = size(df_onehot, 1)
    df_title = deepcopy(df_onehot[!, :Name])
    for i in 1:n
        x1 = findfirst(",", df_onehot[i, :Name])[1] + 2
        x2 = findfirst(".", df_onehot[i, :Name])[1] - 1
        title = df_onehot[i, :Name]
            if title[x1:x2] == "Mlle"
                df_title[i] = "Miss"
            elseif title[x1:x2] == "Jonkheer" || title[x1:x2] == "Don"
                df_title[i] = "Mr"
            elseif title[x1:x2] == "Mme" 
                df_title[i] = "Mrs"
            elseif title[x1:x2] == "Dona" 
                df_title[i] = "Ms"
            else
                df_title[i] = title[x1:x2]
            end
    end
    df_onehot[!, :Title] = df_title
    df_onehot = onehotencoder!(df_onehot, :Title)
    df_onehot[!, :Title] = df_title

    #replace the missing age values
    age_missing = deepcopy(df_onehot.Age)
    group_ages = Dict()
    unique_groups = unique(df_onehot.Title)
    for group in unique_groups
        index = findall(x -> x == group, df_onehot.Title)
        group_ages[group] = mean(skipmissing(df_onehot.Age[index]))
    end

    for i in 1:size(df_onehot, 1)
        if ismissing(df_onehot[i, :Age])
            age_missing[i] = 1
            df_onehot.Age[i] = group_ages[df_onehot.Title[i]]
        else
            age_missing[i] = 0
        end
    end
    
    #Ticket number
    df_nrTicket = []
    strings = df_onehot.Ticket
    for i in 1:length(df_onehot.Ticket)
      push!(df_nrTicket, parse(Float64,split.(strings, " ")[i][end]))
    end
    df_onehot[!, :nrTicket] = df_nrTicket

    #gender
    df_onehot[!, :Gender] = zeros(length(df_onehot.Sex))
    ind = findall(x -> x == "female", df_onehot.Sex)
    df_onehot.Gender[ind] = zeros(length(ind))
    ind = findall(x -> x == "male", df_onehot.Sex)
    df_onehot.Gender[ind] = ones(length(ind))

    #ageband onehot (turned out to be worse)
    #df_onehot[!, :Ageband] = Array{String}(undef, length(df_onehot.Age)) 
    #for i in 1:length(df_onehot.Age)
    #    if df_onehot.Age[i] <= 16
    #        df_onehot[i, :Ageband] = "Child"
    #    elseif df_onehot.Age[i] > 16 && df_onehot.Age[i] <= 36
    #        df_onehot[i, :Ageband] = "Adult"
    #    else df_onehot.Age[i] > 36
    #        df_onehot[i, :Ageband] = "Elder"
    #    end
    #end
    #df_onehot = onehotencoder!(df_onehot, :Ageband) 

    #minmax normalisation of age, ticketnumber and fare
    df_onehot.Age = minmaxnorm(df_onehot.Age)
    df_onehot.nrTicket = minmaxnorm(df_onehot.nrTicket)
    df_onehot.Fare = minmaxnorm(df_onehot.Fare)

    #delete unnecessary columns
    delete!(df_onehot, :Ticket)
    delete!(df_onehot, :PassengerId)
    delete!(df_onehot, :Title)
    delete!(df_onehot, :Sex)
    delete!(df_onehot, :Name)  
    return df_onehot
end

#read in for lars
train = CSV.read("C:/Users/lakad/OneDrive/Documents Uni/BIO397/julia/competition/train.csv", copycols = true)
test = CSV.read("C:/Users/lakad/OneDrive/Documents Uni/BIO397/julia/competition/test.csv", copycols = true)
#read in for jenni
train = CSV.read("C:/Users/probs/Desktop/Sem 7/Machine learning/Files/Competition/titanic/train.csv", copycols = true)
test = CSV.read("C:/Users/probs/Desktop/Sem 7/Machine learning/Files/Competition/titanic/test.csv", copycols = true)
#read in for jonas
train = CSV.read("C:/Users/Jonas/Desktop/Uni/semester_5/Blockkurs_2/titanic_competition/train.csv"; copycols=true)
test = CSV.read("C:/Users/Jonas/Desktop/Uni/semester_5/Blockkurs_2/titanic_competition/test.csv"; copycols=true)

#split in x and y data
train = train[train.Ticket .!== "LINE", :] #split out the 4 wired people
y_train = train.Survived
CSV.write("y_train_onehot.csv", train);
delete!(train, :Survived)
test.Fare[test.PassengerId .== 1044] = [0] #got this value with research

#onehot encode and delete one hot encoded lines
train_onehot= onehot_titanic(train);
names!(train_onehot, [:Age, :SibSp, :Parch, :Fare, :Embarked_C, :Embarked_Q, :Embarked_S, :Embarked_missing, :Pclass_1, :Pclass_2, :Pclass_3, :Pclass_missing, :deck_A, :deck_B, :deck_C, :deck_D, :deck_E, :deck_F, :deck_G, :deck_T, :deck_missing, :Capt, :Col, :Dr, :Lady, :Major, :Master, :Miss, :Mr, :Mrs, :Ms, :Rev, :Sir, :Countess, :Title_missing, :nrTicket, :Gender])

test_onehot = onehot_titanic(test);
names!(test_onehot, [:Age, :SibSp, :Parch, :Fare, :Embarked_C, :Embarked_Q, :Embarked_S, :Embarked_missing, :Pclass_1, :Pclass_2, :Pclass_3, :Pclass_missing, :deck_A, :deck_B, :deck_C, :deck_D, :deck_E, :deck_F, :deck_G, :deck_missing, :Col, :Dr, :Master, :Miss, :Mr, :Mrs, :Ms, :Rev, :Title_missing, :nrTicket, :Gender])
#get the same features in both datasets
insert!(test_onehot, 20, zeros(size(test_onehot, 1)), :deck_T)
insert!(test_onehot, 22, zeros(size(test_onehot, 1)), :Capt)
insert!(test_onehot, 25, zeros(size(test_onehot, 1)), :Lady)
insert!(test_onehot, 26, zeros(size(test_onehot, 1)), :Major)
insert!(test_onehot, 33, zeros(size(test_onehot, 1)), :Sir)
insert!(test_onehot, 34, zeros(size(test_onehot, 1)), :Countess)

mat_test_onehot = Matrix(test_onehot)
mat_train_onehot = Matrix(train_onehot)

#scan for missing values
print([sum(ismissing.(train_onehot[:,i])) for i in 1:length(train_onehot[1, :])]) #no missing values anymore!!!

#save prepared data sets
CSV.write("train_onehot.csv", train_onehot);
CSV.write("test_onehot.csv", test_onehot);

# get the whole onehotencoded dataset for analysis in RDatasets
all_onehot = hcat(y_train, train_onehot)
print(names(all_onehot))
names!(all_onehot, [:Survived, :Age, :SibSp, :Parch, :Fare, :Embarked_C, :Embarked_Q, :Embarked_S, :Embarked_missing, :Pclass_1, :Pclass_2, :Pclass_3, :Pclass_missing, :deck_A, :deck_B, :deck_C, :deck_D, :deck_E, :deck_F, :deck_G, :deck_T, :deck_missing, :Capt, :Col, :Dr, :Lady, :Major, :Master, :Miss, :Mr, :Mrs, :Ms, :Rev, :Sir, :Countess, :Title_missing, :nrTicket, :Gender])

CSV.write("all_onehot.csv", all_onehot);
    

###########################approach 1: K nearest neighbors
#score: 0.77990
distance(p, q) = sqrt(sum((p .- q) .^ 2))

function k_neighbors(k, dat, query) #main function to detect survival
    distancelist = ones(k) .* 100
    survivallist = zeros(k)
    for i in 1:size(dat, 1)
      r = distance(query, dat[i, :])
      if i<k+1
        distancelist[i]=r
        survivallist[i]= y_train[i]
      end
      if r<maximum(distancelist) #find the ones with lowest distance
        a=findfirst(x -> x == maximum(distancelist), distancelist)
        distancelist[a]=r
        survivallist[a]=y_train[i]
      end
    end
    if sum(survivallist)>round(k/2)
      return 1
    else
      return 0
    end
end

#run model with k=sqrt(n)
predictions_k = []
passengerId = test.PassengerId
for i in 1:size(mat_test_onehot, 1)
    query = mat_test_onehot[i, :]
    push!(predictions_k, k_neighbors(10, mat_train_onehot, query))
end

submission_k = DataFrame( PassengerId = test.PassengerId, Survived = predictions_k) #score: 0.77990

sum(predictions_k) #138

#save the predictions
CSV.write("submission_3.csv", submission_k)

      

#############################LogisticRegression from ScikitLearn 
#score: 0.77511
using RDatasets
using Plots
using ScikitLearn
import ScikitLearn: fit!, predict
@sk_import linear_model: LogisticRegression


model_log= LogisticRegression()
fit!(model_log, mat_train_onehot, y_train)
predictions_log = predict(model_log, mat_test_onehot)

sum(predictions_log) #167

submission_log = DataFrame( PassengerId = test.PassengerId, Survived = predictions_log) 
#save the predictions
CSV.write("submission_4.csv", submission_log)


############################# Decision Tree
# Create a tree
using ScikitLearn
using DecisionTree
using ScikitLearn.CrossValidation: cross_val_score
features = deepcopy(mat_train_onehot)
labels = deepcopy(y_train)

# BEST ALGORITHM
#train depth-truncated classifier (add pruning_purity_threshold) 
#score 0.79425
model_dectree = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=1, max_depth=3)
fit!(model_dectree, features, labels)
# apply learned model
predictions_dectree = predict(model_dectree, mat_test_onehot)
# get the probability of each label
predict_proba(model_dectree, mat_test_onehot)
# run n-fold cross validation over 3 CV folds
accuracy = cross_val_score(model_dectree, features, labels, cv=3)

sum(predictions_dectree)
# max_depth=3 -> 165    =4 -> 185            =6 -> 160

submission_dectree = DataFrame( PassengerId = test.PassengerId, Survived = predictions_dectree) 
#save the predictions
CSV.write("submission_14.csv", submission_dectree)


###### decision tree classifier
# train full-tree classifier 
#score: 0.72727
# build_tree(labels, features, n_subfeatures, max_depth, min_samples_leaf, min_samples_split, min_purity_increase)
model_dt = build_tree(labels, features, 0, -1, 1, 2, 0.0)
# prune tree: merge leaves having >= 90% combined purity (default: 100%)
model_dt = prune_tree(model_dt, 0.9)
# apply learned model
predictions_dt = apply_tree(model_dt, mat_test_onehot)
sum(predictions_dt) # 160
# get the probability of each label
apply_tree_proba(model_dt, mat_test_onehot, [0, 1])
# run 3-fold cross validation of pruned tree
n_folds=3
accuracy = nfoldCV_tree(labels, features, n_folds)

submission_dt = DataFrame( PassengerId = test.PassengerId, Survived = predictions_dt) 
#save the predictions
CSV.write("submission_6.csv", submission_dt)

# set of classification parameters and respective default values
# pruning_purity: purity threshold used for post-pruning (default: 1.0, no pruning)
# max_depth: maximum depth of the decision tree (default: -1, no maximum)
# min_samples_leaf: the minimum number of samples each leaf needs to have (default: 1)
# min_samples_split: the minimum number of samples in needed for a split (default: 2)
# min_purity_increase: minimum purity needed for a split (default: 0.0)
# n_subfeatures: number of features to select at random (default: 0, keep all)



###### Adaptive-Boosted Decision Stumps Classifier
# train adaptive-boosted stumps 
#score: 0.75119
model_ad, coeffs_ad = build_adaboost_stumps(labels, features, 100000);
# apply learned model
predictions_ad = apply_adaboost_stumps(model_ad, coeffs_ad, mat_test_onehot)
sum(predictions_ad) # 100k->152   10k->142    1k->153   100->162    10->125
# get the probability of each label
apply_adaboost_stumps_proba(model_ad, coeffs_ad, mat_test_onehot, [0, 1])
# run 3-fold cross validation for boosted stumps
n_iterations=100_000; n_folds=3
accuracy = nfoldCV_stumps(labels, features, n_folds, n_iterations)

submission_ad = DataFrame( PassengerId = test.PassengerId, Survived = predictions_ad) 
#save the predictions
CSV.write("submission_8.csv", submission_ad)
                          

################################# train random forest classifier
#score: 0.78468
model_rf = build_forest(labels, features, 6, 10, 0.7, 6, 1, 2, 0)
# apply learned model
predictions_rf = apply_forest(model_rf, mat_test_onehot)
sum(predictions_rf) #145
# get the probability of each label
apply_forest_proba(model_rf, mat_test_onehot, [0, 1])
# run 3-fold cross validation for forests, using 2 random features per split
n_folds=3; n_subfeatures=2
accuracy = nfoldCV_forest(labels, features, n_folds, n_subfeatures)

#save the predictions
submission_rf = DataFrame( PassengerId = test.PassengerId, Survived = predictions_rf) 
CSV.write("submission_9.csv", submission_rf)

# set of classification parameters and respective default values
# n_subfeatures: number of features to consider at random per split (default: -1, sqrt(# features))
# n_trees: number of trees to train (default: 10)
# partial_sampling: fraction of samples to train each tree on (default: 0.7)
# max_depth: maximum depth of the decision trees (default: no maximum)
# min_samples_leaf: the minimum number of samples each leaf needs to have (default: 5)
# min_samples_split: the minimum number of samples in needed for a split (default: 2)
# min_purity_increase: minimum purity needed for a split (default: 0.0)



#rename data
features_train = deepcopy(mat_train_onehot)
features_test = deepcopy(mat_test_onehot)
labels=deepcopy(y_train)



############################ Neural network 
#score: 0.76555
using Flux
using Flux: onehotbatch, onecold
using Flux.Tracker: update!, param, back!, grad
using ScikitLearn
using Statistics: mean

## Load the data
@sk_import model_selection: train_test_split


## Split train and test
nsamples, nfeatures = size(mat_train_onehot) # 887x37
X_train, X_test, y_train, y_test = train_test_split(mat_train_onehot, labels, train_size=0.81, test_size=0.19, random_state=3, stratify=labels)
X_train_t = X_train'
X_test_t = X_test'

## One hot encode y
y_train =  Int32.(onehotbatch(y_train, 0:1))
y_test =  Int32.(onehotbatch(y_test, 0:1))

## Define activation functions
reluu(X) = max(0, X)
sigma(X) =  max(min(1 / (exp.(-X) + 1), 0.99999), 0.00001)

reluu(x::T) where T<:Real = max(T(0), x)
sigma(x::T) where T<:Real = max(min(T(1) / (T(1) + exp(-x)), T(0.99999)), T(0.00001))

## Define the model
linear(X, W, b) = W * X .+ b
model1(X, W1, b1, W2, b2, W3, b3) = sigma.(linear(reluu.(linear(reluu.(linear(X, W1, b1)), W2, b2)), W3, b3))

## Define cost function
cost(ypred::Real, y::Real) = (-y * log10(ypred)) - ((eltype(y)(1)-y)*log10(eltype(ypred)(1)-ypred))

function cost(X, Y, W1, b1, W2, b2, W3, b3)
  ypred = model1(X, W1, b1, W2, b2, W3, b3);
  return mean(cost.(ypred, Y))
end

"gradien descent"
function GD(X, Y, epochs, α)
  W1 = param(randn(20, nfeatures) .* 0.01);
  b1 = param(zeros(20, 1));
  W2 = param(randn(15, 20) * 0.01);
  b2 = param(zeros(15, 1));
  W3 = param(randn(2, 15) * 0.01);
  b3 = param(zeros(2, 1));
  for i in 1:epochs
    back!(cost(X, Y, W1, b1, W2, b2, W3, b3))  # Use the back! function for back propogation.
    update!(W1, -α*grad(W1));  # after using the back! function, gradients can be accessed with the grad function.
    update!(b1, -α*grad(b1));
    update!(W2, -α*grad(W2));
    update!(b2, -α*grad(b2));
    update!(W3, -α*grad(W3));
    update!(b3, -α*grad(b3));
  end
  c = cost(X, Y, W1, b1, W2, b2, W3, b3)
  return W1, b1, W2, b2, W3, b3, c
end

α=0.5
# Train the network
W1, b1, W2, b2, W3, b3, c = GD(X_train_t, y_train, 2000, α)
# X_train: 37x718     y_train:718

## Make predictions
ypreds = model1(X_test_t, W1, b1, W2, b2, W3, b3);

## check model accuracy on test part
accuracy(y_test, ypreds) = mean(onecold(y_test) .== onecold(ypreds))
accuracy(y_test, ypreds)

#predict on test data
X_test_t2 = mat_test_onehot'
predictions_nn = model1(X_test_t2, W1, b1, W2, b2, W3, b3);


predictions_nn = onecold(predictions_nn).-1

sum(predictions_nn)

#save the predictions
submission_nn = DataFrame( PassengerId = test.PassengerId, Survived = predictions_nn) 
CSV.write("submission_19.csv", submission_nn)



############################# Gridsearch on best algorithm with same parameter setting
# score is worse: 0.77511
# Create a tree
using ScikitLearn
using Plots
import ScikitLearn: fit!, predict
using ScikitLearn.CrossValidation: cross_val_score
@sk_import neighbors: KNeighborsClassifier
@sk_import model_selection: train_test_split
@sk_import model_selection: GridSearchCV
@sk_import preprocessing: RobustScaler
@sk_import model_selection: StratifiedKFold
@sk_import tree: DecisionTreeClassifier


# Define parameter sets to check
parameters = Dict("min_samples_split" => 2:10, "min_samples_leaf" => 1:15, "max_depth"=> 3:20)

# stratified  K-fold 
kf = StratifiedKFold(n_splits=10, shuffle=true)

# Run the model
dt = DecisionTreeClassifier()
model_dectree = GridSearchCV(dt, parameters, scoring="accuracy", cv=kf, error_score="raise")
fit!(model_dectree, features_train, labels)

# apply learned model
predictions_dectree = predict(model_dectree, features_test)
# run n-fold cross validation over 3 CV folds
accuracy = mean(cross_val_score(model_dectree, features_train, labels, cv=3)) #0.8309054817529393

# Get the estimator
best_estimator = model_dectree.best_estimator_ #min_samples_leaf=10, min_samples_split=8, max_depth=8
best_score = model_dectree.best_score_ #0.8365276211950394
best_estimator = model_dectree.best_params_ 
model_dectree.cv_results_

sum(predictions_dectree) #152

#save the predictions
submission_dectree = DataFrame( PassengerId = test.PassengerId, Survived = predictions_dectree) 
CSV.write("submission_17.csv", submission_dectree)


###############################same with random forest with gridsearch
#worse than without: 0.77511
@sk_import ensemble: RandomForestClassifier

# Define parameter sets to check
parameters = Dict("may_depth" => 3:20, "min_samples_split" => 2:10, "min_samples_leaf" => 1:15)

# stratified  K-fold 
kf = StratifiedKFold(n_splits=10, shuffle=true)

# Run the model
dt = RandomForestClassifier()
model_rf = GridSearchCV(dt, parameters, scoring="accuracy", cv=kf, error_score="raise")
fit!(model_rf, features_train, labels)

# apply learned model
predictions_rf = predict(model_rf, features_test)
# run n-fold cross validation over 3 CV folds
accuracy = cross_val_score(model_rf, features_train, labels, cv=3)

# Get the estimator
best_estimator = model_dectree.best_estimator_ #min_samples_leaf=10, min_samples_split=8, may_depth=8
best_score = model_dectree.best_score_ #0.8365276211950394
best_estimator = model_dectree.best_params_
model_dectree.cv_results_

sum(predictions_rf) #154

#save the predictions
submission_rf = DataFrame( PassengerId = test.PassengerId, Survived = predictions_rf) 
CSV.write("submission_18.csv", submission_rf)