function normalSqEucLB(D, X, r, kfunc, hyp, Xu, m, v, vn, mU0, vU0;
    salt = 1e-5,
    Kuu = 0, ki = 0, kii = 0, Kuuinv = 0, kiKuuinv = 0)

    # compute the means and variances of latent coordinates
    nX = size(X,1)
    mZ = zeros(nX, r)
    vZ = zeros(nX, r)
    nXu = size(Xu,1)

    if Kuu == 0
        Kuu = zeros(nXu, nXu,r)
        for l=1:r
            Kuu[:,:,l] = kfunc(hyp, Xu) + salt*eye(nXu)
        end
    end

    if Kuuinv == 0
        Kuuinv = zeros(nXu, nXu, r)
        for l=1:r
            Kuuinv[:,:,l] = inv(Kuu[:,:,l]) # simple way to replace later
        end
    end

    if ki == 0
        ki = zeros(nX, nXu, r)
        for l=1:r
            for i=1:nX
                ki[i,:,l] = kfunc(hyp, X[i,:], Xu)
            end
        end
    end

    if kii == 0
        kii = zeros(nX, r)
        for l=1:r
            for i=1:nX
                kii[i, l] = (kfunc(hyp, X[i,:]))[1]
            end
        end
    end

    if kiKuuinv == 0
        kiKuuinv = zeros(nX, nXu, r)
        for l=1:r
            for i=1:nX
                kiKuuinv[i,:,l] = ki[i,:,l]*Kuuinv[:,:,l]
            end
        end
    end


    mhatkl = zeros(eltype(m),nX,r)
    vhatkl = zeros(eltype(m),nX,r)

    for l=1:r
        for i=1:nX
            vhatkl[i,l] = kii[i,l] - (kiKuuinv[i,:,l]*diagm(v[:,l])*(kiKuuinv[i,:,l]'))[1]
            vhatkl[i,l] = max(1e-4, vhatkl[i,l])
            mhatkl[i,l] = (kiKuuinv[i,:,l]*m[:,l])[1]
        end
    end


    # compute A1
    nD = size(D,1)
    A1 = 0
    for k=1:nD
        dk = D[k,3]
        i = Int(D[k,1])
        j = Int(D[k,2])

        mzk = (mhatkl[i,:] - mhatkl[j,:]).^2
        vzk = vhatkl[i,:] + vhatkl[j,:]

        Eyk = sum(mzk .+ vzk)

        Eyk2 = 4*sum(mzk.*vzk) + 2*sum(vzk.^2) + Eyk^2
        A1 += dk^2 - 2*dk*Eyk + Eyk2
    end
    dkl = 0.0 #DKLNormDiag(mU, vU, mU0, vU0)
    lb = -nD*log(vn)/2 - A1/(2*vn) - dkl
    lb = -lb;
    return (lb, A1, dkl)
end
