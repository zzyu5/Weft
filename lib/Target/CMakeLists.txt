add_library(WeftRISCVTarget STATIC
  RISCVTargetProfile.cpp
  RISCVLowering.cpp
)

target_link_libraries(WeftRISCVTarget PUBLIC
  WeftExtensionDialect
  WeftKernelDialect
  MLIRIR
)
