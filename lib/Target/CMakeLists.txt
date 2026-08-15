add_library(WeftRISCVTarget STATIC
  RISCVTargetProfile.cpp
  RISCVHeader.cpp
  RISCVLowering.cpp
)

target_link_libraries(WeftRISCVTarget PUBLIC
  WeftExtensionDialect
  WeftKernelDialect
  MLIRIR
)
