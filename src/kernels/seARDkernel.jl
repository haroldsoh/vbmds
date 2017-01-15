function seARDkernel(params, X)
    ls = params.^2
    nX = size(X,1)
    K = zeros(eltype(params[1]), nX,nX)
    for i=1:nX
        for j=1:nX
            K[i,j] = exp(-sum(((X[i,:] - X[j,:]).^2)./ls))[1]
        end
    end
    return K
end

function seARDkernel(params, x, X)
    ls = params.^2
    nX = size(X,1)
    K = zeros(eltype(params[1]), 1,nX)
    for j=1:nX
        K[1,j] = exp(-sum(((x[:] - X[j,:]).^2)./ls))[1]
    end
    return K
end


function softplus(x)
    return log(1.0 + exp(x))
end

function seNNkernel(params, X)
    W = params[1:end-1]
    nD = Int(sqrt(length(W)))
    W = reshape(W, nD, nD )
    ls = params[end].^2
    nX = size(X,1)
    K = zeros(eltype(ls), nX,nX)
    # println(W)
    # println(X)
    for i=1:nX
        for j=1:nX
            #K[i,j] = exp(-sum((( tanh(W*X[i,:]) - tanh(W*X[j,:]) ).^2)./ls))[1]
            #K[i,j] = exp(-sum((( softplus(W*X[i,:]) - softplus(W*X[j,:]) ).^2)./ls))[1]
            K[i,j] = exp(-sum((( (W*X[i,:]) - (W*X[j,:]) ).^2)./ls))[1]
        end
    end
    return K
end

function seNNkernel(params, x, X)
    W = params[1:end-1]
    nD = Int(sqrt(length(W)))
    W = reshape(W, nD, nD )
    ls = params[end].^2
    nX = size(X,1)
    K = zeros(eltype(params[1]), 1,nX)
    #println(W)
    #println(x)
    for j=1:nX
        #K[1,j] = exp(-sum(((  tanh(W*x[:]) - tanh(W*X[j,:]) ).^2)./ls))[1]
        #K[1,j] = exp(-sum(((  softplus(W*x[:]) - softplus(W*X[j,:]) ).^2)./ls))[1]
        K[1,j] = exp(-sum(((  (W*x[:]) - (W*X[j,:]) ).^2)./ls))[1]
    end
    return K
end


function seEmbkernel(params, X)
    W = params[1:end-1]
    nD = Int(sqrt(length(W)))
    W = reshape(W, nD, nD )
    ls = params[end].^2
    nX = size(X,1)
    K = zeros(eltype(ls), nX,nX)
    for i=1:nX
        for j=1:nX
            #K[i,j] = exp(-sum((( tanh(W*X[i,:]) - tanh(W*X[j,:]) ).^2)./ls))[1]
            K[i,j] = exp(-sum((( W*X[i,:] - W*X[j,:] ).^2)./ls))[1]
        end
    end
    return K
end

function seEmbkernel(params, x, X)
    W = params[1:end-1]

    nD = Int(sqrt(length(W)))
    W = reshape(W, nD, nD )
    ls = params[end].^2
    nX = size(X,1)
    K = zeros(eltype(params[1]), 1,nX)
    for j=1:nX
        #K[1,j] = exp(-sum(((  tanh(W*x[1,:]) - tanh(W*X[j,:]) ).^2)./ls))[1]
        K[1,j] = exp(-sum(((  W*x[:] - W*X[j,:] ).^2)./ls))[1]
    end
    return K
end
