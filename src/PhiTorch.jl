#entry point for julia PhiTorch project
#To execute file, run command: julia --project=. src/PhiTorch.jl

module PhiTorch
include("tensor/MyTensorPkg.jl")
using StaticArrays
using Metal
using .TensorPkg

greet(name) = println("Hello $name \n")
function main()
    println("Hello World")
    v = SVector{3, Float64}(1.0, 2.0, 3.0)
    tens = Tensor(v)
    println(tens.data)
end

main()


# function gpu_dot(tens)
#     print(typeof(tens)) 
# end

end
