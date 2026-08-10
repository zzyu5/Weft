add_library(WeftRISCVTarget STATIC
  RISCVTargetProfile.cpp
  SourceEmitter.cpp
)

target_link_libraries(WeftRISCVTarget PUBLIC
  WeftExecutionDialect
  WeftKernelDialect
  MLIRIR
)
