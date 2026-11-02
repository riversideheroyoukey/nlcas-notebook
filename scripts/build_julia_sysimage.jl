using Pkg

# Build tool lives in the default environment; the APP packages resolve from the
# repo-local project (Project.toml / Manifest.toml) via the `project` kwarg below,
# so the sysimage matches the pinned environment the app actually runs under.
println("Ensuring PackageCompiler is installed...")
Pkg.add("PackageCompiler")
using PackageCompiler

using Libdl
sysimage_ext = Libdl.dlext

repo_root    = normpath(joinpath(@__DIR__, ".."))
sysimage_path = joinpath(repo_root, "notebook", "runtime", "sysimage.$sysimage_ext")

# Must mirror the app's runtime imports (_julia_driver.jl). JSON3 is the driver's
# serialization layer — omitting it here forces a runtime load from outside the
# sysimage, which is exactly what pinning is meant to prevent.
packages = [:QuantumAlgebra, :Symbolics, :Latexify, :SymPy,
            :SymbolicNumericIntegration, :JSON3]

# Make sure the repo project is instantiated before baking it into the sysimage.
Pkg.activate(repo_root)
Pkg.instantiate()
Pkg.activate()  # back to the default env (where PackageCompiler lives)

println("Creating sysimage at $sysimage_path ...")
println("Precompiling packages: ", join(packages, ", "))

PackageCompiler.create_sysimage(
    packages;
    sysimage_path=sysimage_path,
    project=repo_root,
    precompile_execution_file=joinpath(@__DIR__, "sysimage_precompile.jl"),
)
