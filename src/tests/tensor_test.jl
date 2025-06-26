using Pkg
Pkg.activate(@__DIR__)    # enter your project environment
using Test

# load your wrapper module
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using CTensor
println("→ Running all test sets…")

# include every test file under ops-tests/
for dir in filter(isdir, readdir("tests/ops-tests", join=true))
    for tf in filter(f -> endswith(f, ".jl"), readdir(dir, join=true))
        println("\n=== ", basename(tf), " ===")
        include(tf)
    end
end

println("\nAll done.")
end