module CTensor
using Libdl

const libtensor = Libdl.dlopen(joinpath(@__DIR__, "..", "..", "build", "tensorlib.so"))

function dot()
    @assert length(a) == length(b) "vectors must be same length"
    n = length(a)
    ccall((:dot, libtensor), Float32,
          (Ptr{Float32}, Ptr{Float32}, Csize_t),
          a, b, Csize_t(n))
end

end