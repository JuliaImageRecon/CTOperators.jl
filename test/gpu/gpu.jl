# GPU tests. Included by test/gpu/<backend>.jl, which loads the GPU package and defines
# `arrayTypes` (GPU array types) and `floatTypes` (precisions the device supports).

using CTOperators, Test

@testset "GPU $arrayType" for arrayType in arrayTypes
  @testset "$T" for T in floatTypes
    # operator contract tests on arrayType{T} go here
  end
end
