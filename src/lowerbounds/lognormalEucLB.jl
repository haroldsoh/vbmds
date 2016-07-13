function lognormalEucLB(D, X, r, kfunc, hyp, Xu, m, v, vn, mU0, vU0;
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
    kuKuuinvS = zeros(eltype(m), nX, nXu, r)
    S = Array{Diagonal}(r)

    for l=1:r
        S[l] = Diagonal(vec(v[:,l]))
        for i=1:nX
            vhatkl[i,l] = kii[i,l] - (kiKuuinv[i,:,l]*(ki[i,:,l]'))[1]
            vhatkl[i,l] = max(1.0e-4, vhatkl[i,l])
            mhatkl[i,l] = (kiKuuinv[i,:,l]*m[:,l])[1]
            kuKuuinvS[i,:,l] = kiKuuinv[i,:,l]*S[l]
        end
    end


    # compute A1
    nD = size(D,1)
    A1 = 0.0
    lambdak = zeros(eltype(m),r,1)
    for k=1:nD
        dk = D[k,3]
        i = Int(D[k,1])
        j = Int(D[k,2])

        betak = mhatkl[i,:] - mhatkl[j,:]
        betak = betak'
        Sigmak = Diagonal(vec(vhatkl[i,:] + vhatkl[j,:]))

        for l=1:r
          bkl = (kiKuuinv[i,:,l] - kiKuuinv[j,:,l])'
          lambdak[l,:] = (kuKuuinvS[i,:,l]-kuKuuinvS[j,:,l])*bkl;
        end
        Lambdak = Diagonal(vec(lambdak))

        mk = betak'*betak + trace(Lambdak) + trace(Sigmak)
        sk = 4*betak'*(Sigmak*Lambdak*Sigmak + Lambdak )*betak + 2*trace((Lambdak*Sigmak)^2) + 2*trace(Lambdak^2) + 2*trace(Sigmak^2)

        A1 += (log(dk) - log(mk)/2)^2 + sk*(2*log(dk) - log(mk)+1)/(2*mk^2)

    end
    dkl = DKLNormDiag(m, v, mU0, vU0)
    lb = -nD*log(vn)/2 - A1[1]/(2*vn) - dkl
    lb = -lb;
    return (lb, A1, dkl)
end
