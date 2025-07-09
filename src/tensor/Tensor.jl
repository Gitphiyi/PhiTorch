module TensorPkg
using StaticArrays, LinearAlgebra
export Tensor, gpu_dot

    struct Tensor{N, T, S<:StaticVector{N, T} }
        ndim::UInt64
        dSize::UInt64
        data::S
        # grad::S
        # shape::NTuple{N, UInt}
        # stride::NTuple{N, UInt}
        # device::Symbol
    end
    function Tensor(v::SVector{N,T}) where {N,T}
        n = UInt(N)                 
        return Tensor{N,T,typeof(v)}(n, n, v)
    end
    
    function gpu_dot(t1, t2)
        return dot(t1.data, t2.data)
    end

end