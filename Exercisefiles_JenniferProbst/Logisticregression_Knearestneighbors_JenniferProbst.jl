#Regression data plot with R^2
using ScikitLearn
using CSV
using Statistics
using Random
using Plots

@sk_import datasets: load_digits

function gradient_log(X, y, A) #logistic gradient function
  n = size(X, 1)
  predicted = predict_log(X, A)
  res = (y .- predicted)
  newA=deepcopy(A)
  newA[1]=0
  newA = -(1/length(y)) .* (res' * X)'
  return newA
end

function GD_log(X, y, A, epochs, α) #logistic gradient descent function
  for i in 1:epochs
    grada = gradient_log(X, y, A)
    A = A .- (α .* grada)
  end
  return A
end

#R^2 Accuracy
accuracy(ypred, y) = mean(ypred .== y)

#logistic prediction and cost functions
predict_log(X, A) = max.(min.(1 ./ (1 .+ exp.(-X*A)), 0.99999), 0.00001)
cost_log(X, y, A) = (-1/size(y)) * (y' * log.(predict_log(X, A)) + (- y.+1)' * log.(-predict_log(X, A) .+ 1))

#run
digits = load_digits()
X = digits["data"]
y = digits["target"]
n = length(y)
r = ones(size(X,1))
X=Matrix(hcat(r,X))
nfeatures = size(X, 2)
A = rand(nfeatures)
α = 0.001

#-----------------------------------------------------------------------------------------------------------------------------#
#predictions run on the whole set as training set

#create new vectors
newys=[]
for i in 0:9
    y0= zeros(Int64, length(y))
    y0[findall(x -> x==i, y)] .= 1
    push!(newys, y0)
end

#create model for each of new vectors
avec=[]
for i in 1:10
  Ai = GD_log(X, newys[i], A, 1000, α)
  push!(avec, Ai)
end

#predict individual lines, get 10 values (A0 to A9) per line
pred=[]
for i in 1:10
  push!(pred, predict_log(X, avec[i]))
end

#choose top one, save in vector
predictions=[]
for i in 1:length(pred[1])
  compare=[]
  for j in 1:10
    push!(compare, pred[j][i])
  end
  push!(predictions, (findmax(compare)[2])-1)
end

#get accuracy
acc=accuracy(predictions, y) #0.9042849

#----------------------------------------------------------------------------------------------------------------------------#
#Same with Training and Testing data 

function traintest(X, y, perc) #split in train and test dataset
  ran=shuffle(Vector(1:length(y)))
  choose=ran[Int(round(perc*length(y))):end]
  choosenot=ran[1:Int(floor(0.2*length(y)))]
  X_train=Matrix(X[choose,:])
  y_train=y[choose, :]
  X_test=Matrix(X[choosenot,:])
  y_test=y[choosenot, :]
  return X_train, X_test, y_train, y_test
end

function settrainingsize(X_train, y_train, siz) #set the size of trainingset
  return Matrix(X_train[1:siz, :]), y_train[1:siz, :]
end

function kfold(X, y, k) #split in k folds and output sets to be tested
  dif=Int(round(length(y)/k))
  ran=Vector(1:length(y))
  Xvec=[]
  yvec=[]
  for i in 1:k
    if i == k
      choose=ran[(i-1)*dif+1:end]
    else
      choose=ran[(i-1)*dif+1:i*dif]
    end
    X_set=Matrix(X[Not(choose),:])
    push!(Xvec, X_set)
    y_set=y[Not(choose),:]
    push!(yvec, y_set)
  end
  return Xvec, yvec
end

function learningcurve(X, y, A, perc, k) #run the training on different trainingsetsizes and plot learningcurve
  X_train, X_test, y_train, y_test = traintest(X, y, perc) #separate test set
  trainingsizes=collect(k:100:size(X_train)[1]) #list of trainingsizes
  learn_trainscores_persize = []
  learn_testscores_persize = []
  for siz in trainingsizes #loop for all the trainingsizes
    learn_trainscores = []
    learn_testscores = []
    Xtrain, ytrain = settrainingsize(X_train, y_train, siz)
    Xfolds, yfolds = kfold(Xtrain, ytrain, k)
    for i in 1:k #loop through the folds to train
      trX = Xfolds[i]
      trY = yfolds[i]
      nfeatures = size(trX, 2)
      A = rand(nfeatures)
      newys = []
      for i in 0:9
        y0= zeros(Int64, length(trY))
        y0[findall(x -> x==i, trY)] .= 1
        push!(newys, y0)
      end
      #create model for each of new vectors
      avec=[]
      for i in 1:10
        Ai = GD_log(trX, newys[i], A, 1000, α)
          push!(avec, Ai)
      end
      #predict for training dataset
      pred=[]
      for i in 1:10
        push!(pred, predict_log(X_train, avec[i]))
      end
      #choose top one, save in vector
      predictions_train=[]
      for i in 1:length(pred[1])
        compare=[]
        for j in 1:10
          push!(compare, pred[j][i])
        end
        push!(predictions_train, (findmax(compare)[2])-1)
      end
      #get accuracy
      acc_train=r2(predictions_train, y_train)
      #predict for test dataset
      pred=[]
      for i in 1:10
        push!(pred, predict_log(X_test, avec[i]))
      end
      #choose top one, save in vector
      predictions_test=[]
      for i in 1:length(pred[1])
        compare=[]
        for j in 1:10
          push!(compare, pred[j][i])
        end
        push!(predictions_test, (findmax(compare)[2])-1)
      end
      #get accuracy
      acc_test=r2(predictions_test, y_test)
      #push values to list
      push!(learn_testscores, acc_test)
      push!(learn_trainscores, acc_train)
    end
    #push means to list per trainingsize
    push!(learn_testscores_persize, mean(learn_testscores))
    push!(learn_trainscores_persize,mean(learn_trainscores))
  end
  return learn_testscores_persize, learn_trainscores_persize, trainingsizes
end

#plot function for learning_curve 
function plot_learning(trsi, trsc, tesc, title="title", xlab="xlab", ylab="ylab", label1="Training accuracy", label2="CV accuracy")
  plot_trsc=mean(trsc, dims=2)
  plot_tesc=mean(tesc, dims=2)
  plot(trsi, plot_trsc, title=title, xlabel=xlab, ylabel=ylab, label=label1)
  plot!(trsi, plot_tesc, label=label2)
end

#set parameters
digits = load_digits()
X = digits["data"]
y = digits["target"]
n = length(y)
r = ones(size(X,1))
X=Matrix(hcat(r,X))
nfeatures = size(X, 2)
A = rand(nfeatures)
α = 0.001
perc=0.2 #how much of data for testing
k=5 #how many folds

#run the model
learn_test_scores, learn_train_scores, train_sizes = learningcurve(X, y, A, perc, k)
plot_learning(train_sizes, learn_train_scores, learn_test_scores, "Learning Curve", "Training size", "Accuracy")

#-------------------------------------------------------------------------------------------------------------------------------#
#K-nearest neighbours and the gender prediction

function zeroone(x) #convert strings of 1 to integers
  if x == '1'
    return 1
  else
    return 0
  end
end

function propertype(query) #get queries and train data into correct format
  alphabet=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
  vowels=["a", "e", "i", "o", "u"]
  hard=["b", "d", "g", "p", "t", "k"]
  indfeat=[]
  #1name
  push!(indfeat, query[1])
  #2gender: 0 für female, 1 für male
  if size(query)>(1,1)
    if string(query[2])=="female"
      push!(indfeat, 0)
    else
      push!(indfeat, 1)
    end
  end
  #3length of name
  push!(indfeat, length(indfeat[1]))
  #4number of vowels
  num=0
  for i in indfeat[1]
    if string(i) in vowels
      num+=1
    end
  end
  push!(indfeat, num)
  #5number of hard consonants
  num=0
  for i in indfeat[1]
    if string(i) in hard
      num+=1
    end
  end
  push!(indfeat, num)
  #6first letter 
  a= findfirst(x -> x == string(indfeat[1][1]), alphabet)
  x = Base.bin(UInt(a), 5, false)
  push!(indfeat, zeroone(x[1]),zeroone(x[2]),zeroone(x[3]),zeroone(x[4]),zeroone(x[5]))
  #7last letter
  b=findfirst(x -> x == string(indfeat[1][end]), alphabet)
  y = Base.bin(UInt(b), 5, false)
  push!(indfeat, zeroone(y[1]),zeroone(y[2]),zeroone(y[3]),zeroone(y[4]),zeroone(y[5]))
  #return array with proper feature format
  return indfeat
end

#distance measure
distance(p, q) = sqrt(sum((p .- q) .^ 2))

function k_neighbors(k, dat, query) #main function to detect gender of name
  distanzliste=ones(k) .* 100
  genderliste=zeros(k)
  query=propertype(query)[2:end]
  for i in 1:size(dat, 1)
    r = distance(query, features[i, 2:end])
    if i<k+1
      distanzliste[i]=r
      genderliste[i]=features[i, 1]
    end
    if r<maximum(distanzliste) #find the ones with lowest distance
      a=findfirst(x -> x == maximum(distanzliste), distanzliste)
      distanzliste[a]=r
      genderliste[a]=features[i, 1]
    end
  end
  if sum(genderliste)>round(k/2) #assign male if more male in the list with close distances
    return "male"
  else
    return "female"
  end
end

#read in data
name = CSV.read("name_gender.txt")
features=zeros(size(name, 1), 14)

#create feature vectores
for i in 1:size(name, 1)
  indfeat = propertype(name[i,:])
  features[i,:]= indfeat[2:end] #push to big array
end

#run model
name= "jenni" #name to change here
k_neighbors(5, features, [name]) 