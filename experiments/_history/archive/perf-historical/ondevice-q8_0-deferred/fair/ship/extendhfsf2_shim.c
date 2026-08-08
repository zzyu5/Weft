// Faithful soft-float __extendhfsf2 shim, ONLY to resolve the libcall the no-Zfh
// kernels emit so the confounded path can be re-measured on-board this session.
// The board's openEuler libgcc was built WITH hardware fp16 and omits this symbol;
// clang's compiler-rt (which a no-Zfh build normally links) provides it. ABI seen
// at the call site (objdump): half bits boxed into an F register (fmv.w.x fa0,a0),
// float returned in fa0 -> signature float(float). Separate TU + noinline => a real
// cross-TU call with the same call/spill overhead as the compiler-rt builtin, which
// IS the confound being measured. Conversion is exact IEEE fp16->fp32 (matches the
// driver's integer path and the pinned oracle).
#include <stdint.h>
#include <string.h>
__attribute__((noinline)) float __extendhfsf2(float a_boxed) {
  uint32_t bits;
  memcpy(&bits, &a_boxed, 4);
  uint16_t h = (uint16_t)bits;
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t man = h & 0x3ffu;
  uint32_t out;
  if (exp == 0) {
    if (man == 0) {
      out = sign;
    } else {
      int e = 127 - 15 + 1;
      while (!(man & 0x400u)) { man <<= 1; e--; }
      man &= 0x3ffu;
      out = sign | ((uint32_t)e << 23) | (man << 13);
    }
  } else if (exp == 0x1f) {
    out = sign | 0x7f800000u | (man << 13);
  } else {
    out = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float f;
  memcpy(&f, &out, 4);
  return f;
}
