module vbmds

# relies on the following packages
using ForwardDiff
using Optim

# file includes
include("src/utils.jl")
include("src/kernels.jl")
include("src/lowerbounds.jl")
include("src/mdsmodels.jl")

# exports
export sekernel, seARDkernel
export normalSqEucLB, lognormalEucLB
export lognormalEucMixLB
export VBMDSGP, VBMDSGPParams
export DKLNormDiag

end
