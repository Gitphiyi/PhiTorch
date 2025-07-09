#entry point for julia PhiTorch project
#To execute file, run command: julia --project=. src/PhiTorch.jl

module PhiTorch
include("tensor/Tensor.jl")
using StaticArrays, Metal, BenchmarkTools
using .TensorPkg

greet(name) = println("Hello $name \n")
function main()
    println("Hello World")
    v = SVector{8, Float64}(1.0, 2.0, 3.0, 4, 5, 6, 7,8)
    tens = Tensor(v)
    tens2 = Tensor(v)
    elapsed = @elapsed res = gpu_dot(tens, tens2)
    println(res)
    println(elapsed)
end

main()

end
