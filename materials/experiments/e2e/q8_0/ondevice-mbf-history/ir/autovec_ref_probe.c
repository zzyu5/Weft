#include <stddef.h>
#include <stdint.h>

#define QK 32
typedef struct __attribute__((packed)) { uint16_t d; int8_t qs[QK]; } block_q8_0;
static float fp16_to_fp32(uint16_t h){
  uint32_t sign=(uint32_t)(h&0x8000u)<<16, exp=(h>>10)&0x1fu, man=h&0x3ffu, out;
  if(exp==0){ if(man==0) out=sign; else { int e=127-15+1; while(!(man&0x400u)){man<<=1;e--;} man&=0x3ffu; out=sign|((uint32_t)e<<23)|(man<<13);} }
  else if(exp==0x1f) out=sign|0x7f800000u|(man<<13);
  else out=sign|((exp-15+127)<<23)|(man<<13);
  float f; __builtin_memcpy(&f,&out,sizeof f); return f;
}
static int32_t ref_block_sumi(const block_q8_0*x,const block_q8_0*y){
  int32_t sumi=0; for(int j=0;j<QK;j++) sumi+=(int32_t)x->qs[j]*(int32_t)y->qs[j]; return sumi;
}
__attribute__((noinline)) float ref_vec_dot(size_t n,const block_q8_0*x,const block_q8_0*y){
  size_t nb=n/QK; float sumf=0.0f;
  for(size_t i=0;i<nb;i++){ int32_t sumi=ref_block_sumi(&x[i],&y[i]);
    float dx=fp16_to_fp32(x[i].d), dy=fp16_to_fp32(y[i].d), dxdy=dx*dy;
    sumf=sumf+(float)sumi*dxdy; }
  return sumf;
}
