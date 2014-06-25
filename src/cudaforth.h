
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <malloc.h>
#include <cuda_runtime.h>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
            file, line );
    exit( EXIT_FAILURE );
  }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

typedef unsigned long long word;

extern __shared__ word STATES[];

#define GTID      (threadIdx.x + blockIdx.x * blockDim.x)
#define TID       threadIdx.x*16

#define SP        STATES[TID]
#define IP        STATES[TID + 1]
#define STACK(sp) STATES[TID + 2 + sp]

#define PUSH(x)   STACK(SP++) = (x)
#define POP()     STACK(--SP)
#define TOS()     STACK[SP]

#define UNBOX(n)  ((n) >> 4)

#define IBOX(n)   ((n) << 4 | 2)
#define IIS(n)    ((n) & 2)

#define OBOX(n)   ((n) << 4 | 4)
#define OIS(n)    ((n) & 4)


enum opcodes {
  // io
  IPRINT, // (i -- ) print integer
  NL,     // ( -- ) print nl

  // flow
  EXIT,   // ( -- ) exit interpreter

  // gpu
  LANEID,
  THREADIDX,
  THREADIDY,
  THREADIDZ,
  BLOCKIDX,
  BLOCKIDY,
  BLOCKIDZ,
  BLOCKDIMX,
  BLOCKDIMY,
  BLOCKDIMZ,

  // memory
  MALLOC,
  PITCH,
  FREE,

  // vote
  ANY,    // (i -- o) non-zero if any of i non-zero
  ALL,    // (i -- o) non-zero if all of i non-zero
  BALLOT, // (i -- o) bit flipped for thread N if i

  // shuffly
  ISHFL,      // (var delta width -- i)
  ISHFL_UP,   // (var delta width -- i)
  ISHFL_DOWN, // (var delta width -- i)
  ISHFL_XOR,  // (var mask width -- i)

  FSHFL,      // (var delta width -- i)
  FSHFL_UP,   // (var delta width -- i)
  FSHFL_DOWN, // (var delta width -- i)
  FSHFL_XOR,  // (var mask width -- i)

  // fence
  THREADFENCE,
  THREADFENCE_BLOCK,
  THREADFENCE_SYSTEM,

  SYNCTHREADS,

  // math
  ADD,
  SUB,
  MUL,
  DIV,
  ACOSF,
  ACOSHF,
  ASINF,
  ATAN2F,
  ATANF,
  ATANHF,
  CBRTF,
  CEILF,
  COPYSIGNF,
  COSF,
  COSHF,
  COSPIF,
  ERFCF,
  ERFCINVF,
  ERFCXF,
  ERFF,
  ERFINVF,
  EXP10F,
  EXP2F,
  EXPF,
  EXPM1F,
  FABSF,
  FDIMF,
  FDIVIDEF,
  FLOORF,
  FMAF,
  FMAXF,
  FMINF,
  FMODF,
  FREXPF,
  HYPOTF,
  ILOGBF,
  ISFINITE,
  ISINF,
  ISNAN,
  J0F,
  J1F,
  JNF,
  LDEXPF,
  LGAMMAF,
  LLRINTF,
  LLROUNDF,
  LOG10F,
  LOG1PF,
  LOG2F,
  LOGBF,
  LOGF,
  LRINTF,
  LROUNDF,
  MODFF,
  NANF,
  NEARBYINTF,
  NEXTAFTERF,
  NORMCDFF,
  NORMCDFINVF,
  POWF,
  RCBRTF,
  REMAINDERF,
  REMQUOF,
  RINTF,
  ROUNDF,
  RSQRTF,
  SCALBLNF,
  SCALBNF,
  SIGNBIT,
  SINCOSF,
  SINCOSPIF,
  SINF,
  SINHF,
  SINPIF,
  SQRTF,
  TANF,
  TANHF,
  TGAMMAF,
  TRUNCF,
  Y0F,
  Y1F,
  YNF,
};

typedef struct Words {word code; char *name;} Words;

static Words CoreWords[] = {
  {ADD, "+"},
  {SUB, "-"},
  {MUL, "*"},
  {DIV, "/"},
  {IPRINT, "iprint"},
  {NL, "nl"},
};
