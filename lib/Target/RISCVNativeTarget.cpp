#include "Weft/Target/RISCVTargetProfile.h"

#if defined(__linux__) && defined(__riscv)
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/RISCVISAInfo.h"

#include <cerrno>
#include <cstring>
#include <sched.h>
#include <sys/auxv.h>
#include <sys/prctl.h>
#endif

bool weft::queryNativeRISCVTarget(RISCVNativeTarget &result,
                                  std::string &error) {
#if defined(__linux__) && defined(__riscv)
  constexpr int getVectorControl = 70; // Linux PR_RISCV_V_GET_CONTROL.
  constexpr unsigned long vectorCapability = 1UL << ('V' - 'A');
  const int control = prctl(getVectorControl, 0UL, 0UL, 0UL, 0UL);
  if (control < 0 || (control & 3) == 1 ||
      !(getauxval(AT_HWCAP) & vectorCapability)) {
    error = "native target requires Linux-enabled vector state";
    return false;
  }
  auto features = llvm::sys::getHostCPUFeatures();
  if (!features.lookup("v")) {
    error = "native target discovery requires hwprobe-confirmed full V support";
    return false;
  }
  std::vector<std::string> enabled;
  for (const auto &feature : features)
    if (feature.second &&
        llvm::RISCVISAInfo::isSupportedExtensionFeature(feature.first()))
      enabled.push_back("+" + feature.first().str());
  auto isa = llvm::RISCVISAInfo::parseFeatures(__riscv_xlen, enabled);
  if (!isa) {
    error = llvm::toString(isa.takeError());
    return false;
  }
  std::string march = "rv" + std::to_string(__riscv_xlen);
  // Preserve the reported extension vocabulary. Implied subextensions are
  // already covered by their parent requirement, not additional build flags.
  for (const auto &extension : (*isa)->getExtensions())
    if (extension.first.size() == 1 && features.lookup(extension.first))
      march += extension.first;
  for (const auto &extension : (*isa)->getExtensions())
    if (extension.first.size() > 1 && features.lookup(extension.first))
      march += "_" + extension.first;

  cpu_set_t original;
  if (sched_getaffinity(0, sizeof(original), &original) != 0) {
    error = "cannot discover native CPU affinity: " + std::string(strerror(errno));
    return false;
  }
  result.cpus.clear();
  uintptr_t bytes = 0;
  bool success = true;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &original))
      continue;
    cpu_set_t selected;
    CPU_ZERO(&selected);
    CPU_SET(cpu, &selected);
    if (sched_setaffinity(0, sizeof(selected), &selected) != 0) {
      error = "cannot bind CPU for VLEN discovery: " + std::string(strerror(errno));
      success = false;
      break;
    }
    uintptr_t currentBytes;
    asm volatile("csrr %0, 0xc22" : "=r"(currentBytes));
    if (!currentBytes || (bytes && bytes != currentBytes)) {
      error = "native CPU affinity does not have one positive VLEN; select a homogeneous CPU set";
      success = false;
      break;
    }
    bytes = currentBytes;
    result.cpus.push_back(cpu);
  }
  if (sched_setaffinity(0, sizeof(original), &original) != 0) {
    error = "cannot restore CPU affinity after VLEN discovery: " + std::string(strerror(errno));
    return false;
  }
  if (!success || result.cpus.empty())
    return false;
  std::string abi = __riscv_xlen == 64 ? "lp64" : "ilp32";
#if defined(__riscv_float_abi_double)
  abi += "d";
#elif defined(__riscv_float_abi_single)
  abi += "f";
#endif
  return parseRISCVTargetProfile(march, abi, bytes * 8, "none", 2,
      "independent-multilevel", "within-record", result.profile, error);
#else
  error = "native target discovery requires a Weft compiler running on RISC-V Linux";
  return false;
#endif
}
