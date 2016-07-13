function sekernel(params, X)
    ls = params[1].^2
    nX = size(X,1)
    K = zeros(nX,nX)
    for i=1:nX
        for j=1:nX
            K[i,j] = exp(-sum((X[i,:] - X[j,:]).^2)./ls)
        end
    end
    return K
end

function sekernel(params, x, X)
    ls = params[1].^2
    nX = size(X,1)
    K = zeros(1,nX)
    for j=1:nX
        K[1,j] = exp(-sum((x[1,:] - X[j,:]).^2)./ls)
    end
    return K
end
