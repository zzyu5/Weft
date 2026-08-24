add_library(WeftRISCVTarget STATIC
  RISCVTargetProfile.cpp
  RISCVPlanningSupport.cpp
  RISCVConstructProblems.cpp
  RISCVConstrainRepresentations.cpp
  RISCVResolveLayoutConversions.cpp
  RISCVPropagateStorageMappings.cpp
  RISCVConstrainInstructions.cpp
  RISCVScheduleLevels.cpp
  RISCVConstrainResources.cpp
  RISCVSolve.cpp
  RISCVIntrinsicC.cpp
  RISCVCompiler.cpp
)

target_link_libraries(WeftRISCVTarget PUBLIC
  WeftKernelDialect
  WeftRISCVDialect
  MLIRIR
  MLIRPass
)
