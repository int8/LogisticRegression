# LogisticRegression
Julia implementation of Logistic Regression. It involves optimization of log-likelihood function using gradient descen method, basic evaluation metrics such as AUC has been implemented too. 

To run basic example try:

```julia
include("LogisticRegression.jl");
include("Evaluation.jl");

# Pkg.add("RDatasets") - install RDatasets if you haven't already 
# Phg.add("Gadfly") - install gadfly for visualization purposes 
using RDatasets
using Gadfly

biopsy  = dataset("MASS","biopsy");
median_of_missing_V6 = int(median(biopsy[:V6][!isna(biopsy[:V6])]))
biopsy[:V6][isna(biopsy[:V6])] = median_of_missing_V6
data = convert(Array{Float64,2}, biopsy[2:10]);
labels = convert(Array{Float64,1}, biopsy[:11] .== "benign");

params = Dict([:alpha, :eps, :max_iter], [0.001, 0.0005, 10000.]);
learn_method = Dict([:learning_method, :params, :inferencer], [logistic_regression_learn, params, predict]);
predictions, eval_results = cross_validation(10, learn_method, data, labels, auc);

plot(x = eval_results[1][:,1], y= eval_results[1][:,2], Geom.line, Guide.xlabel("1 - Specificity"), Guide.ylabel("Sensivitiy"))
```