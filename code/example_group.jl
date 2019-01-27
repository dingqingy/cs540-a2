# Load X and y variable
using JLD
data = load("../data/groupData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit multi-class logistic regression classifer
include("logReg.jl")
# model = logRegSoftmax(X,y)
# model = logRegSoftmaxL2(X,y)
# model = logRegSoftmaxL1(X,y)
model = logRegSoftmaxGL1(X,y)


# Compute training and validation error
using Statistics
yhat = model.predict(X)
trainError = mean(yhat .!= y)
yhat = model.predict(Xtest)
validError = mean(yhat .!= ytest)

# Count number of parameters in model and number of features used
nModelParams = sum(model.w .!= 0)
nFeaturesUsed = sum(sum(abs.(model.w),dims=2) .!= 0)
@show(trainError)
@show(validError)
@show(nModelParams)
@show(nFeaturesUsed)

# Show the image as a matrix
using PyPlot
imshow(model.w);
