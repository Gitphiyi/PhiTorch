# install the package
using Pkg
Pkg.add("Metal")

# smoke test
using Metal
Metal.versioninfo()

print("Hello World")

function gpu_dot()
end