using Distances
using Optim
using ForwardDiff
import Base.copy
using MultivariateStats
using NearestNeighbors

type VBMDSGPParams
    Xu # inducing input locations
    num_lower_dim # dimensions
    lbfunc
    kfunc # kernel function
    hyp # hyperparameters
    m # mean of inducing variables (set to 0 if you want auto init)
    v # diagonal of inducing variables (set to 0 if you want auto init)
    vn #noise variance
    m0 # initial prior for m
    v0 # intiial prior for v
    init_method # "random" for random initialization, "mds" for MDS initialization
    mds_vn # noise for mds method if used

    function VBMDSGPParams( Xu, num_lower_dim,lbfunc,kfunc, hyp,m,v,vn, m0,v0, init_method, mds_vn)
        p = new()
        p.Xu= Xu
        p.num_lower_dim =num_lower_dim
        p.lbfunc = lbfunc
        p.kfunc = kfunc
        p.hyp = hyp
        p.m = m
        p.v = v
        p.vn = vn
        p.m0 = m0
        p.v0 = v0
        p.init_method = init_method
        p.mds_vn = mds_vn
        return p
    end

    function VBMDSGPParams(rhs::VBMDSGPParams)
        p = new()
        p.Xu= rhs.Xu
        p.num_lower_dim = rhs.num_lower_dim
        p.lbfunc = rhs.lbfunc
        p.kfunc = rhs.kfunc
        p.hyp = rhs.hyp
        p.m = rhs.m
        p.v = rhs.v
        p.vn = rhs.vn
        p.m0 = rhs.m0
        p.v0 = rhs.v0
        p.init_method = rhs.init_method
        p.mds_vn = rhs.mds_vn
        return p
    end

end

function copy(rhs::VBMDSGPParams)
    return VBMDSGPParams(rhs)
end

function VBMDSGP(D, X, params::VBMDSGPParams;
    sample_size = 0,
    num_iterations=5,
    cg_iterations=2,
    local_iterations=100,
    report_interval=10,
    fixed_noise = false,
    fixed_hyp = false,
    show_trace = false
    )

    # extract parameters
    const Xu = params.Xu
    const nXu = size(Xu,1)

    const r = params.num_lower_dim
    const kfunc = params.kfunc
    const hyp = params.hyp
    const m = params.m
    const v = params.v
    const vn = params.vn
    const m0 = params.m0
    const v0 = params.v0
    const mds_vn = params.mds_vn
    lbfunc = params.lbfunc

    # dataset parameters
    const Dlocal = D
    const nD = size(Dlocal,1)
    const Xfeat = X
    const nX = size(X,1)

    sampleD = 0
    theta_fixed = 0
    hyp_fixed = 0
    vn_fixed = 0

    # random initialization
    function randomThetaInit()
        return (rand(nXu*r)*3, 0.1*ones(nXu*r))
    end

    # MDS initialization
    function mdsThetaInit()
        Mdist = pairwise(Euclidean(), X').^2
        mds_res = classical_mds(Mdist.^(1/2) + mds_vn*randn(size(Mdist)), r)
        mds_res = mds_res' - repmat(mds_res[:,1]', size(Mdist,1), 1)

        # find closest mds point to each Xu point
        tree = KDTree(X')

        m_init = zeros(size(Xu,1), r)
        for i=1:size(Xu,1)
            ind, dists = knn(tree, vec(Xu[i,:]), 1)
            m_init[i,:] = mds_res[ind[1], :] + 0.1*rand(1,r)
        end
        return (m_init, 0.1*ones(nXu*r))
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

    function optGPFunc(theta)
        m = reshape(theta[1:nXu*r], nXu, r)
        v = reshape(theta[(nXu*r+1):end], nXu, r).^2

        fval, A1, dkl = lbfunc(sampleD, Xfeat, r,
            kfunc, hyp_fixed, Xu, m, v, vn_fixed, m0, v0;
            Kuu = Kuu_fixed, ki = ki_fixed, kii = kii_fixed,
            Kuuinv = Kuuinv_fixed, kiKuuinv = kiKuuinv_fixed)

        return fval
    end


    function optimizeGP(theta_init)
        res = optimize(optGPFunc, theta_init, method =:cg,
        autodiff=true, iterations=num_iterations,
        show_trace=show_trace, ftol=1e-6)
        return res.minimum
    end

    function optimizeNoise(vn_init)

    end

    function optimizeHyp(hyp_init)

    end

    function learn(theta, vn, hyp)
        curr_th = deepcopy(theta)
        curr_vn = deepcopy(vn)
        curr_hyp = deepcopy(hyp)

        prev_rep_itr = 0
        sampleDVal = Dlocal[rand(1:nD, sample_size), :]
        for itr=1:num_iterations
            println("Outer Iteration: ", itr)
            # optimize points
            vn_fixed = curr_vn
            hyp_fixed = curr_hyp
            for itrj = 1:local_iterations
                println("Local Sample Iteration: " , itrj)
                if (sample_size < 1)
                  sampleD = Dlocal
                else
                  sampleD = Dlocal[rand(1:nD, sample_size), :]
                end
                res = optimize(optGPFunc, curr_th, method =:cg, autodiff=true,
                show_trace =show_trace, ftol=1e-6, iterations=cg_iterations)
                curr_th = res.minimum
            end

            # show validation error
            sampleD = Dlocal[rand(1:nD, sample_size), :]
            fval = optGPFunc(curr_th)
#             res = optimize(optNoise, [curr_th[end]], method =:cg, autodiff=true,
#             show_trace =false, ftol=1e-6, iterations=2)
#             curr_th[end]= res.minimum[1]

            # optimize hyperparameters

            # optimize the noise param
            #vn_opt = optimizeNoise(vn)

            itr += 1
            println(itr, ": ", fval)
        end
        return (curr_th, vn, hyp)
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
                #vhatkl[i,l] = kii[i,l] - (kiKuuinv[i,:,l]*diagm(v[:,l])*(kiKuuinv[i,:,l]'))[1]
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
        if m == 0 || v == 0
            if params.init_method == "random"
                (m, v) = randomThetaInit()
            else
                (m, v) = mdsThetaInit()
            end
        end

        #theta_init = vec([ eta1; eta2])
        theta_init = vec([m[:]; v[:]])

        if m0 == 0
          m0 = zeros(size(m[:]))
        end
        if v0 == 0
          v0 = ones(size(v[:]))
        end


        # Do Learning (Optimization)
        # setup the fixed values
        hyp_init = hyp
        vn_init = vn
        (Kuu_fixed, Kuuinv_fixed, ki_fixed, kii_fixed, kiKuuinv_fixed) =
        setupFixedKernelVars(hyp_init)

        theta_opt, vn_opt, hyp_opt = learn(theta_init, vn_init, hyp_init)

        # generate new parameters
        params_opt = VBMDSGPParams(params)
        params_opt.m = reshape(theta_opt[1:nXu*r], nXu, r)
        params_opt.v = reshape(theta_opt[(nXu*r+1):end], nXu, r).^2
        params_opt.vn = vn_opt

        return params_opt
    end
end
