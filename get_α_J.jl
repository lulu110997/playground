# get_α_J.jl
import StaticArrays as sa
import DifferentiableCollisions as dc

function get_α_J(rs::Vector{Float64}, qs::Vector{Float64})
    # set the position and orientation
    ellipsoid.r = sa.SVector{3}(rs[1:3])
    ellipsoid.q = sa.SVector{4}(qs[1:4])
    polygon.r = sa.SVector{3}(rs[4:6])
    polygon.q = sa.SVector{4}(qs[5:8])

    # compute the minimum uniform scaling factor and Jacobian
#     α, x_int, J = dc.proximity_jacobian(ellipsoid, polygon; verbose=false, pdip_tol=1e-6)
#     return α, x_int, J

    α, J = dc.proximity_gradient(ellipsoid, polygon; verbose=false, pdip_tol=1e-6)
    return α, J
end
