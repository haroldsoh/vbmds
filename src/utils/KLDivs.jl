function DKLNormDiag(mZ, vZ, mZ0, vZ0; return_gradient = false)
    dkl = 0.5*( sum(vZ[:]./vZ0[:]) + sum( ((mZ0[:]- mZ[:]).^2)./vZ0[:])
        - length(mZ[:]) + sum(log(vZ0[:])) - sum(log(vZ[:])))

    grads = ()
    if (return_gradient)
      ddkldmZ = -(mZ0[:] - mZ[:])./vZ0[:]
      ddkldmZ = reshape(ddkldmZ, size(mZ,1), size(mZ,2))
      ddkldvZ = 0.5*( 1./vZ0[:] - 1./vZ[:])
      ddkldvZ = reshape(ddkldvZ, size(vZ,1), size(vZ,2))
      grads = (ddkldmZ, ddkldvZ)
    end

    return (dkl, grads)
end
