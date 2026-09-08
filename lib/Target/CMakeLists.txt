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
  Partial/Support.cpp
  Partial/AddTrees.cpp
  Partial/PlanTopologies.cpp
  Partial/Materialize.cpp
  FinalizeRISCVLeaves.cpp
  SelectRISCVScalarLoadPrimes.cpp
  MaterializeRISCVReadSnapshots.cpp
  MaterializeRISCVResources.cpp
  FuseRISCVPhysicalIssueLoops.cpp
  EliminateDeadRISCVLayouts.cpp
  VerifyFinalRISCV.cpp
  RISCVIntrinsicC.cpp
  Emission/Control.cpp
  Emission/Layout.cpp
  Emission/Memory.cpp
  Emission/RVVArithmetic.cpp
  Emission/Encoding.cpp
  Emission/RVVMemory.cpp
  Emission/IME.cpp
  IME/FragmentEmission.cpp
  IME/FragmentMaterialization.cpp
  Emission/RVVBitplanes.cpp
  Emission/RVVContraction.cpp
  Emission/RVVPartial.cpp
  Emission/Support.cpp
  RISCVCompiler.cpp
)

target_link_libraries(WeftRISCVTarget PUBLIC
  WeftRISCVContracts
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
