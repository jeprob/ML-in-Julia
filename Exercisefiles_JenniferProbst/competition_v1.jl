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
test.Fare[test.PassengerId .== 1044] = [1]

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

    #minmax normalisation of age, ticketnumber and fare
    df_onehot.Age = minmaxnorm(df_onehot.Age)
    df_onehot.nrTicket = minmaxnorm(df_onehot.nrTicket)
    df_onehot.Fare = minmaxnorm(df_onehot.Fare)
    return df_onehot
end

#onehot encode and delete one hot encoded lines
train_onehot= onehot_titanic(train);
delete!(train_onehot, :Ticket)
delete!(train_onehot, :PassengerId)
delete!(train_onehot, :Title)
delete!(train_onehot, :Sex)
delete!(train_onehot, :Name)
names!(train_onehot, [:Age, :SibSp, :Parch, :Fare, :Embarked_C, :Embarked_Q, :Embarked_S, :Embarked_missing, :Pclass_1, :Pclass_2, :Pclass_3, :Pclass_missing, :deck_A, :deck_B, :deck_C, :deck_D, :deck_E, :deck_F, :deck_G, :deck_T, :deck_missing, :Capt, :Col, :Dr, :Lady, :Major, :Master, :Miss, :Mr, :Mrs, :Ms, :Rev, :Sir, :Countess, :Title_missing, :nrTicket, :Gender])

test_onehot = onehot_titanic(test);
delete!(test_onehot, :Ticket)
delete!(test_onehot, :PassengerId)
delete!(test_onehot, :Title)
delete!(test_onehot, :Sex)
delete!(test_onehot, :Name)
names!(test_onehot, [:Age, :SibSp, :Parch, :Fare, :Embarked_C, :Embarked_Q, :Embarked_S, :Embarked_missing, :Pclass_1, :Pclass_2, :Pclass_3, :Pclass_missing, :deck_A, :deck_B, :deck_C, :deck_D, :deck_E, :deck_F, :deck_G, :deck_missing, :Col, :Dr, :Master, :Miss, :Mr, :Mrs, :Ms, :Rev, :Title_missing, :nrTicket, :Gender])
#get the same features in both datasets
insert!(test_onehot, 20, zeros(size(test_onehot, 1)), :deck_T)
insert!(test_onehot, 22, zeros(size(test_onehot, 1)), :Capt)
insert!(test_onehot, 25, zeros(size(test_onehot, 1)), :Lady)
insert!(test_onehot, 26, zeros(size(test_onehot, 1)), :Major)
insert!(test_onehot, 33, zeros(size(test_onehot, 1)), :Sir)
insert!(test_onehot, 34, zeros(size(test_onehot, 1)), :Countess)

#scan for missing values
print([sum(ismissing.(train_onehot[:,i])) for i in 1:length(train_onehot[1, :])]) #no missing values anymore!!!

#save prepared data sets
CSV.write("train_onehot.csv", train_onehot);
CSV.write("test_onehot.csv", test_onehot);
    

#approach 1: K nearest neighbors
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
mat_test_onehot = Matrix(test_onehot)
mat_train_onehot = Matrix(train_onehot)
predictions_k = []
passengerId = test.PassengerId
for i in 1:size(mat_test_onehot, 1)
    query = mat_test_onehot[i, :]
    push!(predictions_k, k_neighbors(10, mat_train_onehot, query))
end

submission_k = DataFrame( PassengerId = test.PassengerId, Survived = predictions_k) #score: 0.77990


#save the predictions
CSV.write("submission_3.csv", submission_k)

      

#LogisticRegression from ScikitLearn 
using RDatasets
using Plots
using ScikitLearn
import ScikitLearn: fit!, predict
@sk_import linear_model: LogisticRegression


model_log= LogisticRegression()
fit!(model_log, mat_train_onehot, y_train)
predictions_log = predict(model_log, mat_test_onehot)

submission_log = DataFrame( PassengerId = test.PassengerId, Survived = predictions_log) 
#save the predictions
CSV.write("submission_4.csv", submission_log)