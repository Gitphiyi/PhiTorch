module CTensor
using Libdl

export dot

const tensorlib = Libdl.dlopen(joinpath(@__DIR__, "..", "..", "build", "tensorlib.so"))

function dot()
    @assert length(a) == length(b) "vectors must be same length"
    print(tensorlib)
end

end