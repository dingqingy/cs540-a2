include("misc.jl")
include("findMin.jl")


# Multi-class softmax version (assumes y_i in {1,2,...,k})
function logRegSoftmax(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	funObj(w) = softmaxObj(w,X,y,k)

	W[:] = findMin(funObj,W[:],derivativeCheck=true,maxIter=500)

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end

function softmaxObj(w,X,y,k)
	(n,d) = size(X)

	W = reshape(w,d,k)

	XW = X*W
	Z = sum(exp.(XW),dims=2)

	nll = 0
	G = zeros(d,k)
	for i in 1:n
		nll += -XW[i,y[i]] + log(Z[i])

		pVals = exp.(XW[i,:])./Z[i]
		for c in 1:k
			G[:,c] += X[i,:]*(pVals[c] - (y[i] == c))
		end
	end
	return (nll,reshape(G,d*k,1))
end


### Softmax L2 START ###
function softmaxObjL2(w,X,y,k,lambda)
	(n,d) = size(X)

	W = reshape(w,d,k)

	XW = X*W
	Z = sum(exp.(XW),dims=2)

	nll = 0
	G = zeros(d,k)
	for i in 1:n
		nll += -XW[i,y[i]] + log(Z[i])

		pVals = exp.(XW[i,:])./Z[i]
		for c in 1:k
			G[:,c] += X[i,:]*(pVals[c] - (y[i] == c))
		end
	end
	nll += 0.5*lambda*norm(W)^2
	G += lambda*W
	return (nll,reshape(G,d*k,1))
end

function logRegSoftmaxL2(X,y,lambda=10)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	funObj(w) = softmaxObjL2(w,X,y,k,lambda)

	W[:] = findMin(funObj,W[:],derivativeCheck=true,maxIter=50)

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end
### Softmax L2 END ###


### Softmax L1 START ###
function logRegSoftmaxL1(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	funObj(w) = softmaxObj(w,X,y,k)

	W[:] = findMinL1(funObj,W[:],10)

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end
### Softmax L1 END ###


### Group L1 START ###
function logRegSoftmaxGL1(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	funObj(w) = softmaxObj(w,X,y,k)

	W[:] = proxGradGroupL1(funObj,W[:],10, d, k, maxIter=30)

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end
### Group L1 END ###