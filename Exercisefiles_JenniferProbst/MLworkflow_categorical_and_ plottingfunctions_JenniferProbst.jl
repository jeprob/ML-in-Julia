#Sample workflow with included plotting functions for learning curves and validation curves

#Categorical data plot with accuracy
using ScikitLearn
using Statistics
using Plots
import ScikitLearn: fit!, predict
@sk_import neighbors: KNeighborsClassifier
@sk_import model_selection: train_test_split
@sk_import model_selection: GridSearchCV
@sk_import preprocessing: RobustScaler
@sk_import model_selection: StratifiedKFold
@sk_import datasets: load_digits

# Load data
digits = load_digits();
X = digits["data"];
y = digits["target"];


# Separate test and training sets
train_size_fraction = 0.71
test_size_fraction = 1 - train_size_fraction
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size_fraction, test_size=test_size_fraction, random_state=3, stratify=y)

# Feature scaling
rscale = RobustScaler()
fit_transform!(rscale, X_train)
fit_transform!(rscale, X_test)

# Define parameter sets to check
parameters = Dict("n_neighbors" => 1:2:30, "weights" => ("uniform", "distance"))

# stratified  K-fold 
kf = StratifiedKFold(n_splits=10, shuffle=true)

# Run the model
knn = KNeighborsClassifier()
model = GridSearchCV(knn, parameters, scoring="accuracy", cv=kf)
fit!(model, X_train, y_train)

# Get the estimator
best_estimator = model.best_estimator_
best_score = model.best_score_
best_estimator = model.best_params_
model.cv_results_

# Make predictions
newy = predict(model, X_test)


# Save the model for future use
using JLD
JLD.save("my_knn.jld", "knn", model)


# Plots
@sk_import model_selection: validation_curve
@sk_import model_selection: learning_curve  

train_sizes, learn_train_scores, learn_test_scores = learning_curve(model, X_train, y_train, cv=kf, scoring="accuracy", shuffle=true)
#train_scores, test_scores = validation_curve(model, X_train, y_train, param_name="n_neighbors", param_range=1:5, cv=kf, scoring="accuracy") #does not work

#plot for learning_curve and validation_curve
function plot_learning(trsi, trsc, tesc, title="title", xlab="xlab", ylab="ylab", label1="Training accuracy", label2="CV accuracy")
    plot_trsc=mean(trsc, dims=2)
    plot_tesc=mean(tesc, dims=2)
    plot(trsi, plot_trsc, title=title, xlabel=xlab, ylabel=ylab, label=label1)
    plot!(trsi, plot_tesc, label=label2)
end

function plot_validation(trsc, tesc, title="title", xlab="xlab", ylab="ylab", label1="", label2="")
    plot_trsc=mean(trsc, dims=2)
    plot_tesc=mean(tesc, dims=2)
    plot(1:5, plot_trsc, seriestype=:scatter, title=title, xlabel=xlab, ylabel=ylab, label=label1)
    plot!(1:5, plot_tesc, label=label2)
end

plot_learning(train_sizes, learn_train_scores, learn_test_scores, "Learning Curve", "Training size", "Accuracy")
#plot_validation(train_scores, test_scores, "Validation Curve", "Parameter values", "Accuracy")