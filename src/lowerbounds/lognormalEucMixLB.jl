using Base.Test



function lognormalEucMixLB(D, X, r,
  kfunc, hyp,
  Xu,
  m, v,
  mu, s2,
  vn,
  mU0, vU0,
  mZ0, vZ0;
  return_gradient = false,
  salt = 1e-5,
  Kuu = 0, ki = 0, kii = 0, Kuuinv = 0, kiKuuinv = 0)

    # compute the means and variances of latent coordinates
    nX = size(X,1)
    nXu = size(Xu,1)
    nZq = size(mu,1)
    nZ = nZq + nX

    # construct if necessary
    if nXu > 0
      if Kuu == 0
          Kuu = zeros(eltype(hyp), nXu, nXu,r)
          for l=1:r
              Kuu[:,:,l] = kfunc(hyp, Xu) + salt*eye(nXu)
          end
      end

      if Kuuinv == 0
          Kuuinv = zeros(eltype(hyp),nXu, nXu, r)
          for l=1:r
              Kuuinv[:,:,l] = inv(Kuu[:,:,l]) # simple way to replace later
          end
      end

      if ki == 0
          ki = zeros(eltype(hyp),nX, nXu, r)
          for l=1:r
              for i=1:nX
                  ki[i,:,l] = kfunc(hyp, X[i,:], Xu)
              end
          end
      end

      if kii == 0
          kii = zeros(eltype(hyp),nX, r)
          for l=1:r
              for i=1:nX
                  kii[i, l] = (kfunc(hyp, X[i,:]))[1]
              end
          end
      end

      if kiKuuinv == 0
          kiKuuinv = zeros(eltype(hyp),nX, nXu, r)
          for l=1:r
              for i=1:nX
                  kiKuuinv[i,:,l] = ki[i,:,l]*Kuuinv[:,:,l]
              end
          end
      end
    end

    # loop through D to see if we can skip any
    valids = Set{Int32}()
    for k = 1:size(D,1)
      push!(valids,D[k,1])
      push!(valids,D[k,2])
    end


    if nXu > 0
      vartype =eltype(m)
    else
      vartype = eltype(mu)
    end
    #check if optimizing hyperparams
    if eltype(hyp) != Float64
        vartype = eltype(hyp)
    end

    mhatkl = zeros(vartype,nZ,r)
    vhatkl = zeros(vartype,nZ,r)
    kiKuuinvS = zeros(vartype, nX, nXu, r)
    S = Array{Diagonal}(r)

    #println("mhat and vhats")
    for l=1:r
      if nX > 0
        S[l] = Diagonal(vec(v[:,l]))
      end
      for i=1:nZ
        if in(i, valids)
          if i <= nX
            # for the GPs
            vhatkl[i,l] = kii[i,l] - (kiKuuinv[i,:,l]*(ki[i,:,l]'))[1]
            vhatkl[i,l] = max(1e-5, vhatkl[i,l])
            mhatkl[i,l] = (kiKuuinv[i,:,l]*m[:,l])[1]
            kiKuuinvS[i,:,l] = kiKuuinv[i,:,l]*S[l]
            #kiKuuinvS[i,:,l] = kiKuuinv[i,:,l].*v[:,l]
          else
            # for the z distributions
            mhatkl[i,l] = mu[i-nX,l]

            vhatkl[i,l] = s2[i-nX,l]
            vhatkl[i,l] = max(1e-5, vhatkl[i,l]) # prevent numerical problems
          end
        end
      end
    end

    # preallocation for gradients
    if return_gradient
      if nZq > 0
        dA1dmu = zeros(vartype, size(mu))
        dA1ds2 = zeros(vartype, size(s2))
      else
        dA1dmu = []
        dA1ds2 = []
      end

      if nXu > 0
        dA1dm = zeros(vartype, size(m))
        dA1dv = zeros(vartype, size(v))
      else
        dA1dm = []
        dA1dv = []
      end
    end


    # compute A1
    nD = size(D,1)
    A1 = 0.0
    lambdak = zeros(vartype,r,1)
    #println("looping through data")
    for k=1:nD
        dk = D[k,3]

        i = Int(D[k,1])
        j = Int(D[k,2])

        betak = mhatkl[i,:] - mhatkl[j,:]
        betak = betak'
        Sigmak = Diagonal(vec(vhatkl[i,:] + vhatkl[j,:]))

        mk = 0.0;
        sk = 0.0;
        if (i > nX && j > nX)
          # both q(Z) points
          mk = betak'*betak + trace(Sigmak)
          sk = 4*betak'*Sigmak*betak + 2*trace(Sigmak.^2)

          # derivatives
          if return_gradient
            dmkdbetak = 2*betak
            #dmkdSigmak = 1.0

            dmkdmui = dmkdbetak
            dmkdmuj = -dmkdbetak
            #dmkds2i = 1.0 #dmkdSigmak
            #dmkds2j = 1.0 #dmkdSigmak

            dskdbetak = 8*Sigmak*betak
            dskdSigmak = 4*(betak.^2) + 4*diag(Sigmak)

            dskdmui = dskdbetak
            dskdmuj = -dskdbetak
            dskds2i = dskdSigmak
            dskds2j = dskdSigmak
          end
        else
          if (i <= nX && j <= nX)
            # both GP points
            qzsign = 0.0 # indicates no qzpoint
            bkl = reshape(kiKuuinv[i,:,:] - kiKuuinv[j,:,:], nXu, r)
            for l=1:r
              lambdak[l,:] = (kiKuuinvS[i,:,l] - kiKuuinvS[j,:,l])*bkl[:,l]
            end

            if return_gradient
              dbetakdm = bkl
              dLambdakdv = (bkl.^2)
            end
          else
            # only one is the GP point
            # first we get which one is the GP point
            # and put the appropriate signs
            gidx = i
            qidx = j
            gpsign = 1.0
            qzsign = -1.0
            if (j<= nX)
              gidx = j
              qidx = i
              gpsign = -1.0
              qzsign = 1.0
            end

            # compute the necessary values for lambda
            bkl = reshape(kiKuuinv[gidx,:,:], nXu, r)
            for l=1:r
              lambdak[l,:] = kiKuuinvS[gidx,:,l]*bkl[:,l]
            end

            if return_gradient
              dbetakdm = gpsign*bkl
              dbetakdmu = qzsign

              dLambdakdv = (bkl.^2)
            end

          end

          # here, either 1 or both are GP points

          Lambdak = Diagonal(vec(lambdak))
          # mk = E[y]
          mk = betak'*betak + trace(Lambdak) + trace(Sigmak)

          # sk = V[y]
          SLk = Sigmak*Lambdak
          SLSk = SLk*Sigmak
          Ellk = SLSk + Lambdak
          sk = 4*betak'*Ellk*betak + 2*trace(SLk^2) + 2*trace(Lambdak^2) + 2*trace(Sigmak^2)

          # gradient computations
          if return_gradient

            # first we handle the GPs (always necessary)
            dmkdbetak = 2.0*betak
            #dmkdLambdak = 1.0
            #dmkdSigmak = 1.0

            dmkdm = zeros(size(m))
            dmkdv = zeros(size(v))
            for l=1:r
              dmkdm[:,l] = dmkdbetak[l]*dbetakdm[:,l]
              dmkdv[:,l] = dLambdakdv[:,l]
            end

            dskdbetak = 8*Ellk*betak
            dskdLamdak =  4*((Sigmak*betak).^2 + betak.^2 + diag(SLSk)) + 4*diag(Lambdak)
            #dskdSigmak = 8*SLk*(betak.^2) + 4*(SLk*Lamdak) + 2*Sigmak

            dskdm = zeros(size(m))
            dskdv = zeros(size(v))
            for l=1:r
              dskdm[:,l] = dskdbetak[l].*dbetakdm[:,l]
              dskdv[:,l] = dskdLamdak[l].*dLambdakdv[:,l]
            end

            # next, we handle the qz point (if necessary)
            if qzsign != 0.0
              dmkdmu = dmkdbetak*dbetakdmu
              #dmkds2 = 1.0

              dskdmu = dskdbetak*dbetakdmu
              dskds2 = 8*SLk*(betak.^2) + diag(4*(SLk*Lambdak) + 2*Sigmak)
            end
          end #end gradient computations
        end # end points, GP only, mixed block selection

        logdk = log(dk)
        logmk = log(mk)

        A1 += (logdk - logmk/2)^2 + sk*(2*logdk - logmk+1)/(2*mk^2)

        # for returning derivaties
        if return_gradient
          dA1dmk = (-sk/(2*mk^3) -sk*(1+2*logdk-logmk)/(mk^3) - (logdk - 0.5*logmk)/mk)[1]
          dA1dsk = ((1 + 2*logdk - logmk)/(2*(mk^2)))[1]

          if (i > nX && j > nX)
            dA1dmu[i-nX,:] += (dA1dmk*dmkdmui + dA1dsk*dskdmui)'
            dA1dmu[j-nX,:] += (dA1dmk*dmkdmuj  + dA1dsk*dskdmuj)'

            dA1ds2[i-nX,:] += (dA1dmk + dA1dsk*dskds2i)'
            dA1ds2[j-nX,:] += (dA1dmk + dA1dsk*dskds2j)'

          elseif (i <= nX && j <= nX)
            dA1dm += dA1dmk*dmkdm + dA1dsk*dskdm
            dA1dv += dA1dmk*dmkdv + dA1dsk*dskdv
          else
            dA1dm += dA1dmk*dmkdm + dA1dsk*dskdm
            dA1dv += dA1dmk*dmkdv + dA1dsk*dskdv

            dA1dmu[qidx-nX,:] += (dA1dmk*dmkdmu + dA1dsk*dskdmu)'
            dA1ds2[qidx-nX,:] += (dA1dmk + dA1dsk*dskds2)'
          end
        end
    end
    #println("done with data")
    # KL Divergence
    dkl = 0.0
    dklgpgrads = (0,0)
    dklqzgrads = (0,0)

    if (nXu > 0)
      dklgp, dklgpgrads = DKLNormDiag(m, v, mU0, vU0; return_gradient=return_gradient)
      dkl += dklgp
    end

    if (size(mu,1) > 0)
      dklqz, dklqzgrads = DKLNormDiag(mu, s2, mZ0, vZ0; return_gradient=return_gradient)
      dkl += dklqz
    end

    # compute final lower bound
    lb = -nD*log(vn)/2 - A1[1]/(2*vn) - dkl
    lb = -lb;

    grads = ()
    if return_gradient
      dlbdm = dA1dm/(2*vn)  + dklgpgrads[1]
      dlbdv = dA1dv/(2*vn) + dklgpgrads[2]
      dlbdmu = dA1dmu/(2*vn) + dklqzgrads[1]
      dlbds2 = dA1ds2/(2*vn) + dklqzgrads[2]
      grads = (dlbdm, dlbdv, dlbdmu, dlbds2) # to restore to dlb later
    end

    return (lb, A1[1], dkl, grads)
end

function lognormalEucMixLB(A1, dkl, nD, vn)
  lb = -nD*log(vn)/2 - A1/(2*vn) - dkl
  return -lb;
end
