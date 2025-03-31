# create_shapes.jl
import StaticArrays as sa
import DifferentiableCollisions as dc

function create_shapes()
    a = 0.06
    b = 0.04
    c = 0.1

    P = sa.@SMatrix [1/(a*a) 0.0 0.0
                    0.0 1/(b*b) 0.0
                    0.0 0.0 1/(c*c)]
    global ellipsoid = dc.Ellipsoid(P)

    A = sa.@SMatrix [1.0 0.0 0.0
                    0.0 1.0 0.0
                    0.0 0.0 1.0
                    -1.0 0.0 0.0
                    0.0 -1.0 0.0
                    0.0 0.0 -1.0]
    b = sa.@SVector [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    global polygon = dc.Polytope(A, b)
end
