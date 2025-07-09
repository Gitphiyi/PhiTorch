module TensorPkg
using StaticArrays
export Tensor

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
    
    function gpu_dot(t1::Tensor, t2::Tensor)
        dot(t1, t2)
    end

end