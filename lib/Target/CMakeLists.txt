add_library(WeftRISCVTarget STATIC
  RISCVTargetProfile.cpp
  RISCVPhysicalSupport.cpp
  ConvertWeftToRISCV.cpp
  SelectRISCVOperations.cpp
  PropagateRISCVLayouts.cpp
  PlanRISCVMemory.cpp
  CanonicalizeRISCVLayouts.cpp
  FuseRISCVBitplanes.cpp
  LowerRISCVComposites.cpp
  HoistRISCVLoopInvariants.cpp
  ScheduleRISCVLevels.cpp
  PipelineRISCVLevels.cpp
  UnrollRISCVLevels.cpp
  ShareRISCVLayeredWindows.cpp
  MaterializeRISCVPartialAccumulators.cpp
  FinalizeRISCVLeaves.cpp
  MaterializeRISCVResources.cpp
  EliminateDeadRISCVLayouts.cpp
  VerifyFinalRISCV.cpp
  RISCVIntrinsicC.cpp
  RISCVCompiler.cpp
)

target_link_libraries(WeftRISCVTarget PUBLIC
  WeftKernelDialect
  WeftRISCVDialect
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransforms
)
