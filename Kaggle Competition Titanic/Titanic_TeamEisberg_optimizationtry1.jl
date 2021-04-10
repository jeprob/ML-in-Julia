#tried to optimize decision trees with different classification --> but classification worse

#imports 
using DataFrames
using CSV
using Statistics
using LinearAlgebra

#read in packages
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
delete!(train, :Survived)
test.Fare[test.PassengerId .== 1044] = [0] #got this value with research

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
function onehot_titanic(df_onehot)
    #embarked
    df_onehot = onehotencoder!(df_onehot, :Embarked)    
    names!(df_onehot, [:PassengerId, :Pclass, :Name, :Sex, :Age, :SibSp, :Parch, :Ticket, :Fare, :Cabin, :Embarked_C, :Embarked_Q, :Embarked_S, :Embarked_missing])
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

    #gender
    df_onehot[!, :Gender] = zeros(length(df_onehot.Sex))
    ind = findall(x -> x == "female", df_onehot.Sex)
    df_onehot.Gender[ind] = zeros(length(ind))
    ind = findall(x -> x == "male", df_onehot.Sex)
    df_onehot.Gender[ind] = ones(length(ind))
    delete!(df_onehot, :Sex)

    #ageband 
    df_onehot[!, :Ageband] = Array{Int64}(undef, length(df_onehot.Age)) 
    for i in 1:length(df_onehot.Age)
       if df_onehot.Age[i] <= 16
           df_onehot[i, :Ageband] = 1 #"Child"
       elseif df_onehot.Age[i] > 16 && df_onehot.Age[i] <= 36
           df_onehot[i, :Ageband] = 2 #"Adult"
       else df_onehot.Age[i] > 36
           df_onehot[i, :Ageband] = 3 #"Elder"
       end
    end
    delete!(df_onehot, :Age)

    #family or alone
    df_onehot[!, :Alone] = Array{Int64}(undef, length(df_onehot.SibSp)) #1 for passengers travelling alone, 0 for passengers with family
    for i in 1:length(df_onehot.SibSp)
       if df_onehot.SibSp[i] == 0 && df_onehot.Parch[i] == 0
           df_onehot[i, :Alone] = 1
       else 
           df_onehot[i, :Alone] = 0
       end
    end

    #fare
    df_onehot[!, :fare] = Array{Int64}(undef, length(df_onehot.Fare))
    for i in 1:length(df_onehot.Fare)
        if df_onehot.Fare[i] <= 7.9
            df_onehot[i, :fare] = 1 #low
        elseif df_onehot.Fare[i] > 7.9 && df_onehot.Fare[i] <= 14.5
            df_onehot[i, :fare] = 2 #medium
         elseif df_onehot.Fare[i] > 14.5 && df_onehot.Fare[i] <= 31
             df_onehot[i, :fare] = 3 #high
        else df_onehot.Fare[i] > 31
            df_onehot[i, :fare] = 4 #really high
        end
    end
    delete!(df_onehot, :Fare)

    df_onehot[!, :title] = Array{Int64}(undef, length(df_onehot.Title))
    for i in 1:length(df_onehot.Title)
        if df_onehot.Title[i] == "Mr"
            df_onehot[i, :title] = 1 
        elseif df_onehot.Title[i]== "Mrs"
            df_onehot[i, :title] = 2 
        elseif df_onehot.Title[i] == "Miss"
             df_onehot[i, :title] = 3
        elseif df_onehot.Title[i] == "Master"
            df_onehot[i, :title] = 4
        elseif df_onehot.Title[i] == "Ms"
            df_onehot[i, :title] = 5
        else df_onehot.Title[i] == "Rev" || df_onehot.Title[i] == "Dr" || df_onehot.Title[i] == "Lady" || df_onehot.Title[i] == "the Countess" || df_onehot.Title[i] == "Capt" || df_onehot.Title[i] == "Col" || df_onehot.Title[i] == "Don" || df_onehot.Title[i] == "Major" || df_onehot.Title[i] == "Sir" || df_onehot.Title[i] == "Jonkheer" || df_onehot.Title[i] == "Donna"    
            df_onehot[i, :title] = 6 #rare group
        end
    end
    delete!(df_onehot, :Title)

    #delete unnecessary features
    delete!(df_onehot, :Ticket)
    delete!(df_onehot, :PassengerId)
    delete!(df_onehot, :Name)

    return df_onehot
end

#onehot encode and delete one hot encoded lines
train_onehot= onehot_titanic(train);
names!(train_onehot, [:Pclass, :SibSp, :Parch, :Embarked_C, :Embarked_Q, :Embarked_S, :Embarked_missing, :deck_A, :deck_B, :deck_C, :deck_D, :deck_E, :deck_F, :deck_G, :deck_T, :deck_missing, :Gender, :Ageband, :Alone, :fare, :title])


test_onehot = onehot_titanic(test);
insert!(test_onehot, 15, zeros(size(test_onehot, 1)), :deck_T)
names!(train_onehot, [:Pclass, :SibSp, :Parch, :Embarked_C, :Embarked_Q, :Embarked_S, :Embarked_missing, :deck_A, :deck_B, :deck_C, :deck_D, :deck_E, :deck_F, :deck_G, :deck_T, :deck_missing, :Title, :Gender, :Ageband, :Alone, :fare])

mat_train_onehot = Matrix(train_onehot)
mat_test_onehot = Matrix(test_onehot)


#scan for missing values 
print([sum(ismissing.(train_onehot[:,i])) for i in 1:length(train_onehot[1, :])]) 

#save prepared data sets
CSV.write("train_onehot.csv", train_onehot);
CSV.write("test_onehot.csv", test_onehot);

# put y back into train_onehot for analysis in R
all_onehot = hcat(y_train, train_onehot)
names!(all_onehot,[:Survived, :Pclass, :SibSp, :Parch, :Embarked_C, :Embarked_Q, :Embarked_S, :Embarked_missing, :deck_A, :deck_B, :deck_C, :deck_D, :deck_E, :deck_F, :deck_G, :deck_T, :deck_missing, :Gender, :Ageband, :Alone, :fare, :title])

CSV.write("all_onehot.csv", all_onehot);
    


############################# Decision Tree
# Create a tree
using ScikitLearn
using DecisionTree
using ScikitLearn.CrossValidation: cross_val_score
features = deepcopy(mat_train_onehot)
labels = deepcopy(y_train)

# train depth-truncated classifier (add pruning_purity_threshold) 
#score 0.78468
model_dectree = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=1, max_depth=3)
fit!(model_dectree, features, labels)
# apply learned model
predictions_dectree = predict(model_dectree, mat_test_onehot)
# get the probability of each label
predict_proba(model_dectree, mat_test_onehot)
# run n-fold cross validation over 3 CV folds
accuracy = cross_val_score(model_dectree, features, labels, cv=3)

sum(predictions_dectree)
# max_depth=3 -> 165   

submission_dectree = DataFrame( PassengerId = test.PassengerId, Survived = predictions_dectree) 
#save the predictions
CSV.write("submission_10.csv", submission_dectree)
                         

################################# train random forest classifier
#score: 0.76076
model_rf = build_forest(labels, features, 4, 10, 0.7, -1, 1, 2, 0)
# apply learned model
predictions_rf = apply_forest(model_rf, mat_test_onehot)
sum(predictions_rf) #145
# get the probability of each label
apply_forest_proba(model_rf, mat_test_onehot, [0, 1])
# run 3-fold cross validation for forests, using 2 random features per split
n_folds=3; n_subfeatures=2
accuracy = nfoldCV_forest(labels, features, n_folds, n_subfeatures)

submission_rf = DataFrame( PassengerId = test.PassengerId, Survived = predictions_rf) 
#save the predictions
CSV.write("submission_11.csv", submission_rf)

# set of classification parameters and respective default values
# n_subfeatures: number of features to consider at random per split (default: -1, sqrt(# features))
# n_trees: number of trees to train (default: 10)
# partial_sampling: fraction of samples to train each tree on (default: 0.7)
# max_depth: maximum depth of the decision trees (default: no maximum)
# min_samples_leaf: the minimum number of samples each leaf needs to have (default: 5)
# min_samples_split: the minimum number of samples in needed for a split (default: 2)
# min_purity_increase: minimum purity needed for a split (default: 0.0)