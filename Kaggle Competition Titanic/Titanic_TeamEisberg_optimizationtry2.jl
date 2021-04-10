# 2nd try to optimize decision trees with gridsearch of parameters --> results worse

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

#onehot encode the parameters --> embarked and SibSp seem to have no impact
function onehot_titanic(df_onehot)
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
    delete!(df_onehot, :Sex)

    #ageband --> turns out to be worse
    # df_onehot[!, :Ageband] = Array{Int64}(undef, length(df_onehot.Age)) 
    # for i in 1:length(df_onehot.Age)
    #    if df_onehot.Age[i] <= 16
    #        df_onehot[i, :Ageband] = 1 #"Child"
    #    elseif df_onehot.Age[i] > 16 && df_onehot.Age[i] <= 36
    #        df_onehot[i, :Ageband] = 2 #"Adult"
    #    else df_onehot.Age[i] > 36
    #        df_onehot[i, :Ageband] = 3 #"Elder"
    #    end
    # end
    # delete!(df_onehot, :Age)

    #family or alone
    df_onehot[!, :Alone] = Array{Int64}(undef, length(df_onehot.SibSp)) #1 for passengers travelling alone, 0 for passengers with family
    for i in 1:length(df_onehot.SibSp)
       if df_onehot.SibSp[i] == 0 && df_onehot.Parch[i] == 0
           df_onehot[i, :Alone] = 1
       else 
           df_onehot[i, :Alone] = 0
       end
    end

    #fare --> turns out to be worse
    # df_onehot[!, :fare] = Array{Int64}(undef, length(df_onehot.Fare))
    # for i in 1:length(df_onehot.Fare)
    #     if df_onehot.Fare[i] <= 7.9
    #         df_onehot[i, :fare] = 1 #low
    #     elseif df_onehot.Fare[i] > 7.9 && df_onehot.Fare[i] <= 14.5
    #         df_onehot[i, :fare] = 2 #medium
    #      elseif df_onehot.Fare[i] > 14.5 && df_onehot.Fare[i] <= 31
    #          df_onehot[i, :fare] = 3 #high
    #     else df_onehot.Fare[i] > 31
    #         df_onehot[i, :fare] = 4 #really high
    #     end
    # end
    # delete!(df_onehot, :Fare)

    #delete unnecessary features
    delete!(df_onehot, :Ticket)
    delete!(df_onehot, :PassengerId)
    delete!(df_onehot, :Name)
    delete!(df_onehot, :Title)
    delete!(df_onehot, :SibSp)
    delete!(df_onehot, :Embarked)

    #minmax normalisation of age, ticketnumber and fare
    df_onehot.Age = minmaxnorm(df_onehot.Age)
    df_onehot.nrTicket = minmaxnorm(df_onehot.nrTicket)
    df_onehot.Fare = minmaxnorm(df_onehot.Fare)
    return df_onehot
end

#read in packages
#read in for lars
#train = CSV.read("C:/Users/lakad/OneDrive/Documents Uni/BIO397/julia/competition/train.csv", copycols = true)
#test = CSV.read("C:/Users/lakad/OneDrive/Documents Uni/BIO397/julia/competition/test.csv", copycols = true)
#read in for jenni
train = CSV.read("C:/Users/probs/Desktop/Sem 7/Machine learning/Files/Competition/titanic/train.csv", copycols = true)
test = CSV.read("C:/Users/probs/Desktop/Sem 7/Machine learning/Files/Competition/titanic/test.csv", copycols = true)
#read in for jonas
#train = CSV.read("C:/Users/Jonas/Desktop/Uni/semester_5/Blockkurs_2/titanic_competition/train.csv"; copycols=true)
#test = CSV.read("C:/Users/Jonas/Desktop/Uni/semester_5/Blockkurs_2/titanic_competition/test.csv"; copycols=true)

#split in x and y data
train = train[train.Ticket .!== "LINE", :] #split out the 4 wired people
y_train = train.Survived
delete!(train, :Survived)
test.Fare[test.PassengerId .== 1044] = [0] #got this value with research


#onehot encode and delete one hot encoded lines
train_onehot= onehot_titanic(train);
print(first(train_onehot,1))
names!(train_onehot, [:Age, :Parch, :Fare, :Pclass_1, :Pclass_2, :Pclass_3, :Pclass_missing, :deck_A, :deck_B, :deck_C, :deck_D, :deck_E, :deck_F, :deck_G, :deck_T, :deck_missing, :Capt, :Col, :Dr, :Lady, :Major, :Master, :Miss, :Mr, :Mrs, :Ms, :Rev, :Sir, :Countess, :Title_missing, :nrTicket, :Gender, :Alone])
#best parameter setting so far

test_onehot = onehot_titanic(test);
names!(test_onehot, [:Age, :Parch, :Fare, :Pclass_1, :Pclass_2, :Pclass_3, :Pclass_missing, :deck_A, :deck_B, :deck_C, :deck_D, :deck_E, :deck_F, :deck_G, :deck_missing, :Col, :Dr, :Master, :Miss, :Mr, :Mrs, :Ms, :Rev, :Title_missing, :nrTicket, :Gender, :Alone])

#get the same features in both datasets
insert!(test_onehot, 14, zeros(size(test_onehot, 1)), :deck_T)
insert!(test_onehot, 16, zeros(size(test_onehot, 1)), :Capt)
insert!(test_onehot, 19, zeros(size(test_onehot, 1)), :Lady)
insert!(test_onehot, 20, zeros(size(test_onehot, 1)), :Major)
insert!(test_onehot, 27, zeros(size(test_onehot, 1)), :Sir)
insert!(test_onehot, 28, zeros(size(test_onehot, 1)), :Countess)

mat_train_onehot = Matrix(train_onehot)
mat_test_onehot = Matrix(test_onehot)

#scan for missing values 
print([sum(ismissing.(train_onehot[:,i])) for i in 1:length(train_onehot[1, :])]) 
print([sum(ismissing.(test_onehot[:,i])) for i in 1:length(test_onehot[1, :])]) 


#save prepared data sets
CSV.write("train_onehot.csv", train_onehot);
CSV.write("test_onehot.csv", test_onehot);

# put y back into train_onehot for analysis in R (not onehotencoded)
all_onehot = hcat(y_train, train_onehot)
print(names(all_onehot))
#names!(all_onehot,[:Survived, :Pclass, :SibSp, :Parch, :Cabin, :Embarked, :Title, :nrTicket, :Gender, :Ageband, :Alone, :fare])
CSV.write("all_onehot.csv", all_onehot);

#############################Models

#rename data
features_train = deepcopy(mat_train_onehot)
features_test = deepcopy(mat_test_onehot)
labels=deepcopy(y_train)


############################# Decision Tree
# Create a tree
#score: 0.78947
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
best_estimator = model_dectree.best_estimator_ #min_samples_leaf=10, min_samples_split=2, max_depth=8
best_score = model_dectree.best_score_ #0.8387824126268321
best_estimator = model_dectree.best_params_ 
model_dectree.cv_results_

sum(predictions_dectree) #162

#save the predictions
submission_dectree = DataFrame( PassengerId = test.PassengerId, Survived = predictions_dectree) 
CSV.write("submission_16.csv", submission_dectree)


###############################same with random forest 
@sk_import ensemble: RandomForestClassifier

# Define parameter sets to check
parameters = Dict("n_estimators" => 10:10:100, "min_samples_split" => 2:10, "min_samples_leaf" => 1:15)

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
best_estimator = model_dectree.best_estimator_ #
best_score = model_dectree.best_score_ #0.830890642615558
best_estimator = model_dectree.best_params_
model_dectree.cv_results_

sum(predictions_rf) #148

#save the predictions
submission_rf = DataFrame( PassengerId = test.PassengerId, Survived = predictions_dectree) 
CSV.write("submission_16.csv", submission_rf)