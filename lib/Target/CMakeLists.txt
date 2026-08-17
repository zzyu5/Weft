add_library(WeftRISCVTarget STATIC
  RISCVAxisMapping.cpp
  RISCVCABI.cpp
  RISCVTargetProfile.cpp
  RISCVHeader.cpp
  RISCVKernelFacts.cpp
  RISCVReuseAnalysis.cpp
  RISCVPhysicalPlanning.cpp
  RISCVRVVSpelling.cpp
  RISCVKernelCompiler.cpp
  RISCVIntrinsicCPrelude.cpp
  RISCVRVVIntrinsicC.cpp
  RISCVQuantIntrinsicC.cpp
  RISCVQuantIQ2IntrinsicC.cpp
  RISCVQuantIQ3IntrinsicC.cpp
  RISCVQuantIQ1IntrinsicC.cpp
  RISCVQuantQ6IntrinsicC.cpp
  RISCVQuantPackedIntrinsicC.cpp
  RISCVQuantPackedI3IntrinsicC.cpp
  RISCVQuantTernaryIntrinsicC.cpp
  RISCVQuantSignedCodebookIntrinsicC.cpp
  RISCVIMEIntrinsicC.cpp
  RISCVLowering.cpp
)

target_link_libraries(WeftRISCVTarget PUBLIC
  WeftExtensionDialect
  WeftKernelDialect
  MLIRIR
)
