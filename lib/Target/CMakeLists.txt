add_library(WeftRISCVTarget STATIC
  RISCVTargetProfile.cpp
  RISCVPlanningSupport.cpp
  RISCVConstructProblems.cpp
  RISCVConstrainRepresentations.cpp
  RISCVConstrainInstructions.cpp
  RISCVConstrainResources.cpp
  RISCVSolve.cpp
  RISCVCompiler.cpp
)

target_link_libraries(WeftRISCVTarget PUBLIC
  WeftKernelDialect
  WeftRISCVDialect
  MLIRIR
  MLIRPass
)
