using Distances
using Optim
using ForwardDiff
import Base.copy
using MultivariateStats
using NearestNeighbors
import ForwardDiff
#import ReverseDiffSource

type VBMDSMixParams
    Xu # inducing input locations
    num_lower_dim::Number # dimensions
    lbfunc::Function
    kfunc::Function # kernel function
    hyp::Array{Float64} # hyperparameters
    m::Array{Float64} # mean of inducing variables (set to 0 if you want auto init)
    v::Array{Float64} # diagonal of inducing variables (set to 0 if you want auto init)
    mu::Array{Float64} # mean of z's
    s2::Array{Float64} # diagonal var of z's

    vn::Float64 #noise variance

    m0::Array{Float64} # initial prior for m
    v0::Array{Float64} # intiial prior for v

    mu0::Array{Float64} # initial prior for mu
    s20::Array{Float64} # initial prior for s2

    init_method::AbstractString # "random" for random initialization, "mds" for MDS initialization
    mds_vn::Float64 # noise for mds method if used

    function VBMDSMixParams( Xu, num_lower_dim,
      lbfunc,
      kfunc,hyp,
      m, v,
      mu, s2,
      vn,
      m0,v0,
      mu0, s20,
      init_method,
      mds_vn)
        p = new()
        p.Xu= Xu
        p.num_lower_dim =num_lower_dim
        p.lbfunc = lbfunc
        p.kfunc = kfunc
        p.hyp = hyp
        p.m = m
        p.v = v
        p.mu = mu
        p.s2 = s2
        p.vn = vn
        p.m0 = m0
        p.v0 = v0
        p.mu0 = mu0
        p.s20 = s20
        p.init_method = init_method
        p.mds_vn = mds_vn
        return p
    end

    function VBMDSMixParams(rhs::VBMDSMixParams)
        p = new()
        p.Xu= rhs.Xu
        p.num_lower_dim = rhs.num_lower_dim
        p.lbfunc = rhs.lbfunc
        p.kfunc = rhs.kfunc
        p.hyp = rhs.hyp
        p.m = rhs.m
        p.v = rhs.v
        p.mu = rhs.mu
        p.s2 = rhs.s2
        p.vn = rhs.vn
        p.m0 = rhs.m0
        p.v0 = rhs.v0
        p.mu0 = rhs.mu0
        p.s20 = rhs.s20
        p.init_method = rhs.init_method
        p.mds_vn = rhs.mds_vn
        return p
    end
end

function copy(rhs::VBMDSMixParams)
    return VBMDSMixParams(rhs)
end

function VBMDSMix(D, X, nZ,
  params::VBMDSMixParams;
  opt_method = :cg,
  sample_size = 0,
  num_iterations=5,
  opt_iterations=2,
  opt_hyp_iterations = 5,
  opt_vn_iterations = 5,
  local_iterations=100,
  report_interval=10,
  fixed_noise = false,
  fixed_hyp = false,
  show_trace = false,
  vlb = 1.0e-4,
  vub = 10.0,
  s2lb = 1.0e-4,
  s2ub = 10.0,
  vnlb = 1.0e-4,
  vnub = 1.0e-2,
  hyplb = sqrt(1.0e-2),
  hypub = sqrt(100.0)
  )

    # extract parameters
    const Xu = params.Xu
    const nXu = size(Xu,1)
    const nX = size(X,1)
    const nZq = nZ - nX

    const r = params.num_lower_dim
    const kfunc = params.kfunc
    const hyp = params.hyp
    const m = params.m
    const v = params.v
    const mu = params.mu
    const s2 = params.s2
    const vn = params.vn
    const m0 = params.m0
    const v0 = params.v0
    const mu0 = params.mu0
    const s20 = params.s20
    lbfunc = params.lbfunc

    # dataset parameters
    const Dlocal = D
    const nD = size(Dlocal,1)
    const Xfeat = X

    sampleD = 0
    theta_fixed = 0
    m_fixed = 0
    v_fixed = 0
    mu_fixed = 0
    s2_fixed = 0
    hyp_fixed = 0
    vn_fixed = 0
    A1_fixed = 0
    dkl_fixed = 0
    # random initialization
    function randomThetaInit()
        return (
        3*randn(nXu*r), # m
        min(vub-0.001, max(vlb+0.001, rand(nXu*r))), #v
        3*randn(nZq*r ), #mu
        min(s2ub-0.001, max(s2lb+0.001, rand(nZq*r ))) #s2
        )
    end

    # MDS initialization
    function mdsThetaInit()
      error("not yet done")
        # Mdist = pairwise(Euclidean(), X').^2
        # mds_res = classical_mds(Mdist.^(1/2) + mds_vn*randn(size(Mdist)), r)
        # mds_res = mds_res' - repmat(mds_res[:,1]', size(Mdist,1), 1)
        #
        # # find closest mds point to each Xu point
        # tree = KDTree(X')
        #
        # m_init = zeros(size(Xu,1), r)
        # for i=1:size(Xu,1)
        #     ind, dists = knn(tree, vec(Xu[i,:]), 1)
        #     m_init[i,:] = mds_res[ind[1], :] + 0.1*rand(1,r)
        # end
        # return (m_init, 0.1*ones(nXu*r))
    end

    function sigmtrans(t, lb, ub; return_gradient=false)
        denom = (1.0 + exp(-t))
        rtio = (ub - lb)./denom
        val = lb + rtio
        if (val == ub)
            val -= 1.0e-7
        elseif val == lb
            val += 1.0e-7
        end
        grad = ()
        if return_gradient
          grad = (denom - 1.0).*rtio./denom
          return (val, grad)
        end
        return val
    end

    function invsigmtrans(s, lb, ub)
        if (s == ub)
            s -= 1.0e-7
        elseif s == lb
            s += 1.0e-7
        end
        return -log(((ub - lb)./(s - lb)) - 1.0 )
    end

    function paramsToTheta(m, v, mu, s2)
        theta = [m[:];
        invsigmtrans(v[:], vlb, vub);
        mu[:];
        invsigmtrans(s2[:], s2lb, s2ub)]
    end

    function gradsToTheta(dm, dv, dmu, ds2, dvtrans, ds2trans)
        theta = [dm[:];
        (dv.*dvtrans)[:];
        dmu[:];
        (ds2.*ds2trans)[:]]
    end

    function thetaToParams(theta; return_gradient=false)
        if nXu > 0
            th1 = theta[1:2*nXu*r]
            m = reshape(th1[1:nXu*r], nXu, r)
            v = reshape(th1[(nXu*r+1):end], nXu, r)
            if return_gradient
                v, vgrad = sigmtrans(v, vlb, vub; return_gradient=true)
            else
                v = sigmtrans(v, vlb, vub)
            end
        else
            m = []
            v = []
            vgrad = []
        end

        if nZq != 0
            th2 = theta[(2*nXu*r+1):end]
            mu = reshape(th2[1:nZq*r], nZq, r)
            s2 = reshape(th2[(nZq*r+1):end], nZq, r)
            if return_gradient
                s2, s2grad = sigmtrans(s2, s2lb, s2ub; return_gradient=true)
            else
                s2 = sigmtrans(s2, s2lb, s2ub)
            end
        else
            mu = []
            s2 = []
            s2grad = []
        end

        if return_gradient
            return (m,v,mu,s2,vgrad,s2grad)
        end
        return (m,v,mu,s2)
    end

    # setup priors
    function initPriors()
      if nXu > 0
        if m0 == []
          m0 = zeros(size(m))
        end
        if v0 == []
          v0 = ones(size(v))
        end
      end
      if nZq > 0
        if mu0 == []
          mu0 = zeros(size(mu))
        end
        if s20 == []
          s20 = 10.0*ones(size(s2))
        end
      else
        mu0 = []
        s20 = []
      end
      return m0, v0, mu0, s20
    end

    # optimization functions
    function setupFixedKernelVars(hyp_fixed)
        Kuu_fixed = zeros(nXu, nXu,r)
        for l=1:r
            Kuu_fixed[:,:,l] = kfunc(hyp_fixed, Xu) + 1e-5*eye(nXu)
        end

        Kuuinv_fixed = zeros(nXu, nXu, r)
        for l=1:r
            Kuuinv_fixed[:,:,l] = inv(Kuu_fixed[:,:,l]) # simple way to replace later
        end

        ki_fixed = zeros(nX, nXu, r)
        for l=1:r
            for i=1:nX
                ki_fixed[i,:,l] = kfunc(hyp_fixed, Xfeat[i,:], Xu)
            end
        end

        kii_fixed = zeros(nX, r)
        for l=1:r
            for i=1:nX
                kii_fixed[i, l] = (kfunc(hyp_fixed, Xfeat[i,:]))[1]
            end
        end

        kiKuuinv_fixed = zeros(nX, nXu, r)
        for l=1:r
            for i=1:nX
                kiKuuinv_fixed[i,:,l] = ki_fixed[i,:,l]*Kuuinv_fixed[:,:,l]
            end
        end

        return (Kuu_fixed, Kuuinv_fixed, ki_fixed, kii_fixed, kiKuuinv_fixed)
    end

    function optGPFunc(theta; return_extras = false)
        # extract parameters
        m,v,mu,s2 = thetaToParams(theta)

        # calculate values
        fval, A1, dkl, grads = lbfunc(sampleD, Xfeat, r,
            kfunc, hyp_fixed, Xu,
            m, v,
            mu, s2,
            vn_fixed,
            m0, v0,
            mu0, s20
            ;
            return_gradient = false,
            Kuu = Kuu_fixed, ki = ki_fixed, kii = kii_fixed,
            Kuuinv = Kuuinv_fixed, kiKuuinv = kiKuuinv_fixed)
        if return_extras
          return (fval, A1, dkl, grads)
        end
        return fval
    end

    function optGPGrad!(theta, storage )
        # extract parameters
        m,v,mu,s2,vgrad,s2grad = thetaToParams(theta, return_gradient=true)

        # calculate values
        fval, A1, dkl, grads = lbfunc(sampleD, Xfeat, r,
            kfunc, hyp_fixed, Xu,
            m, v,
            mu, s2,
            vn_fixed,
            m0, v0,
            mu0, s20
            ; return_gradient=true,
            Kuu = Kuu_fixed, ki = ki_fixed, kii = kii_fixed,
            Kuuinv = Kuuinv_fixed, kiKuuinv = kiKuuinv_fixed)

        # have to do one more derivative computation for s2 and v
        # since we constrain them via sigmtrans
        # println(size(theta))
        storage[:] = gradsToTheta(grads[1], grads[2], grads[3], grads[4], vgrad, s2grad)
        # println(storage)
        return fval
    end

    function optHypFunc(theta_hyp; return_extras = false)
        # extract parameters
        #println(theta_hyp, " ", hyplb, " ", hypub)
        curr_hyp = sigmtrans(theta_hyp, hyplb, hypub)

        # calculate values
        fval, A1, dkl, grads = lbfunc(sampleD, Xfeat, r,
            kfunc, curr_hyp, Xu,
            m_fixed, v_fixed,
            mu_fixed, s2_fixed,
            vn_fixed,
            m0, v0,
            mu0, s20)
        if return_extras
          return (fval, A1, dkl, grads)
        end
        return fval
    end



    function optNoiseFunc(theta_vn)
      vn_local = sigmtrans(theta_vn[1], vnlb, vnub)
      # calculate values
      fval = lbfunc(A1_fixed, dkl_fixed, nD, vn_local)
      return fval
    end

    #g = []
    # function gpgrad!(th, storage)
    #   ForwardDiff.gradient!(th, optGPFunc, storage)
    # end

    function optimizeGP(th)
        use_custom_optimizer = false
        if !use_custom_optimizer
            d4 = DifferentiableFunction(optGPFunc, optGPGrad!, optGPGrad!)
            res = optimize(d4, th, method =opt_method,  iterations=opt_iterations,
            show_trace=show_trace, ftol=1e-6)
            println("Number Function Calls: ", res.f_calls)
            println("Number Gradient Calls: ", res.g_calls)
            return res.minimum
        else
            theta = th
            df = zeros(size(theta))
            #df_ref = zeros(size(theta))
            #g = ForwardDiff.gradient(optGPFunc)
            for i=0:opt_iterations
                fval = optGPGrad!(theta, df)
                #df_ref = g(theta)
                #@test df_ref ≈ df
                normdf = norm(df)
                if (normdf < 1e-7)
                    println("Norm is below 1e-7. Stopping optimization")
                    return theta
                end

                theta -= 0.5*sign(df).*min(abs(df), 0.5)
                if i%report_interval == 0
                    println(i, ": fval: ", fval, " Grad Norm: ", normdf)
                end
            end
            return theta


            # chunk_size = 10
            # for i=10:-1:1
            #   if (length(th)%i == 0)
            #     chunk_size = i
            #     println("Chunk Size: ", chunk_size)
            #     break
            #   end
            # end
            #
            # theta = th
            # println("Fval: ", fval)
            # for i=1:cg_iterations
            #     fval = optGPGrad!(theta,
            #   theta -= 0.5*sign(df).*min(abs(df), 0.5)
            #   if i%report_interval == 0
            #     fval = optGPFunc(theta)
            #     println(i, ": fval: ", fval)
            #   end
            # end
            # return theta
        end
    end

    function optimizeNoise(vnl)
      try
        th_vn = invsigmtrans(vnl, vnlb, vnub)
        res = optimize(optNoiseFunc, collect(th_vn), method=opt_method,
        autodiff=true, iterations=opt_vn_iterations,
        show_trace=show_trace, ftol=1e-6)
        return sigmtrans(res.minimum[1], vnlb, vnub)
      catch err
        println("Noise optimization failed.")
        println(err)
        return vn_init
      end
    end

    function optimizeHyp(hypl)
        try
            th_hyp = invsigmtrans(hypl, hyplb, hypub)
            res = optimize(optHypFunc, collect(th_hyp), method=opt_method,
            autodiff=true, iterations=opt_hyp_iterations,
            show_trace=show_trace, ftol=1e-6)
            return sigmtrans(res.minimum, hyplb, hypub)
        catch err
            println("Hyperparameter optimization failed.")
            println(err)
            return hyp_init
        end
    end

    function learn(theta, vn, hyp)
        curr_th = deepcopy(theta)
        curr_vn = deepcopy(vn)
        curr_hyp = deepcopy(hyp)

        prev_rep_itr = 0
        #sampleDVal = Dlocal[rand(1:nD, sample_size), :]
        for itr=1:num_iterations
            println("Outer Iteration: ", itr)
            # optimize points
            vn_fixed = curr_vn
            hyp_fixed = curr_hyp
            for itrj = 1:local_iterations
                println("Local Sample Iteration: " , itrj)
                if sample_size > 0
                   sampleD = Dlocal[randperm(nD)[1:sample_size], :]
                else
                  sampleD = Dlocal
                end
                prev_th = curr_th
                curr_th = optimizeGP(curr_th)
                if (norm(prev_th - curr_th) < 1e-8)
                    println("Norm < 1e-8. Early finish.")
                    break;
                end
            end

            # show validation error
            #sampleD = Dlocal[rand(1:nD, sample_size), :]
            sampleD = Dlocal
            fval, A1_fixed, dkl_fixed = optGPFunc(curr_th; return_extras = true)

            # fix the required values
            m_fixed, v_fixed, mu_fixed, s2_fixed = thetaToParams(curr_th)
            if (itr < num_iterations)
                # optimize hyperparameters
                if !fixed_hyp
                    vn_fixed = curr_vn
                    prev_hyp = curr_hyp
                    #println("before func: ", curr_hyp)
                    curr_hyp = optimizeHyp(curr_hyp)
                    if (norm(curr_hyp - prev_hyp) > 1e-5)
                        println("Creating new kernel matrices with hyp: ", curr_hyp)
                        (Kuu_fixed, Kuuinv_fixed, ki_fixed, kii_fixed, kiKuuinv_fixed) = setupFixedKernelVars(curr_hyp)
                    end
                    println("Curr hyp: ", curr_hyp)
                end

                # optimize noise
                if !fixed_noise
                    hyp_fixed = curr_hyp
                    curr_vn = optimizeNoise(curr_vn)
                    println("Curr vn: ", curr_vn)
                end
            end

            itr += 1
            println(itr, ": ", fval)
        end
        return (curr_th, curr_vn, curr_hyp)
    end


    function predict()
        nX = size(X,1)
        mZ = zeros(nX, r)
        vZ = zeros(nX, r)
        nXu = size(Xu,1)
        ki = zeros(nX, nXu, r)
        kii = zeros(nX, r)
        Kuu = zeros(nXu, nXu,r)
        Kuuinv = zeros(nXu, nXu, r)
        kiKuuinv = zeros(nX, nXu, r)

        mhatkl = zeros(nX,r)
        vhatkl = zeros(nX,r)
        for l=1:r
            Kuu[:,:,l] = kfunc(hyp, Xu)
            Kuuinv[:,:,l] = inv(Kuu[:,:,l]) # simple way to replace later
        end
        for l=1:r
            for i=1:nX
                kii[i, l] = (kfunc(hyp, X[i,:]))[1]
                ki[i,:,l] = kfunc(hyp, X[i,:], Xu)
                kiKuuinv[i,:,l] = ki[i,:,l]*Kuuinv[:,:,l]
                #vhatkl[i,l] = kii[i,l] - (kiKuuinv[i,:,l]*(ki[i,:,l]'))[1]#(kiKuuinv[i,:,l]*diagm(v[:,l])*(kiKuuinv[i,:,l]'))[1]
                vhatkl[i,l] = kii[i,l] + (kiKuuinv[i,:,l]*diagm(v[:,l])*(kiKuuinv[i,:,l]'))[1]
                vhatkl[i,l] -= (kiKuuinv[i,:,l]*(ki[i,:,l]'))[1]
                mhatkl[i,l] = (kiKuuinv[i,:,l]*m[:,l])[1]
            end
        end
        return (mhatkl, vhatkl)
    end

    if D == 0
        # prediction mode
        return predict()
    else
        # initialize parameters
        if m == []|| v == [] || mu == [] || s2 == []
            if params.init_method == "random"
                (m_t, v_t, mu_t, s2_t) = randomThetaInit()
            else
                (m_t, v_t, mu_t, s2_t) = mdsThetaInit()
            end
            m = (m == []) ? m_t : m
            v = (v == []) ? v_t : v
            mu = (mu == []) ? mu_t : mu
            s2 = (s2 == []) ? s2_t : s2
        end

        #theta_init = vec([ eta1; eta2])
        println("Creating Parameters")
        theta_init = paramsToTheta(m, v, mu, s2)
        m0, v0, mu0, s20 = initPriors()

        # Do Learning (Optimization)
        # setup the fixed values
        hyp_init = hyp
        vn_init = vn
        println("Setting up Fixed Variables")
        (Kuu_fixed, Kuuinv_fixed, ki_fixed, kii_fixed, kiKuuinv_fixed) = setupFixedKernelVars(hyp_init)
        #println(Kuu_fixed)
        # println("Learning")
        theta_opt, vn_opt, hyp_opt = learn(theta_init, vn_init, hyp_init)

        # generate new parameters
        println("Done, returning optimized parameters")
        params_opt = VBMDSMixParams(params)

        params_opt.m, params_opt.v, params_opt.mu, params_opt.s2 = thetaToParams(theta_opt)
        params_opt.vn = vn_opt
        params_opt.hyp = hyp_opt
        return params_opt
    end
end
