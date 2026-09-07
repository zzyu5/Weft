add_library(WeftRISCVTarget STATIC
  RISCVTargetProfile.cpp
  RISCVNativeTarget.cpp
  RISCVPhysicalSupport.cpp
  ConvertWeftToRISCV.cpp
  SelectRISCVOperations.cpp
  PropagateRISCVLayouts.cpp
  PlanRISCVResidency.cpp
  PlanRISCVMemory.cpp
  CanonicalizeRISCVLayouts.cpp
  FuseRISCVBitplanes.cpp
  LowerRISCVComposites.cpp
  MaterializeRISCVPrograms.cpp
  VectorizeRISCVRecordLoops.cpp
  HoistRISCVLoopInvariants.cpp
  ScheduleRISCVLevels.cpp
  PipelineRISCVLevels.cpp
  UnrollRISCVLevels.cpp
  ShareRISCVLayeredWindows.cpp
  MaterializeRISCVPartialAccumulators.cpp
  FinalizeRISCVLeaves.cpp
  SelectRISCVScalarLoadPrimes.cpp
  MaterializeRISCVReadSnapshots.cpp
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

if(LLVM_LINK_LLVM_DYLIB)
  llvm_config(WeftRISCVTarget USE_SHARED TargetParser)
else()
  llvm_config(WeftRISCVTarget TargetParser)
endif()
