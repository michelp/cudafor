#include "cudaforth.h"


__device__ void interpreter(word* program)
{
  IP = 2;
  SP = 0;
  word opcode;
  word cont = 1;
  
  do
    {
      opcode = program[IP++];
      
      if(OIS(opcode))
        {
          opcode = UNBOX(opcode);
          switch(opcode)
            {
            case ADD:
              PUSH(POP() + POP());
              break;
            case SUB:
              PUSH(POP() - POP());
              break;
            case MUL:
              PUSH(POP() * POP());
              break;
            case DIV:
              PUSH(POP() / POP());
              break;
            case IPRINT:
              printf("%i", POP());
              break;
            case NL:
              printf("\n");
              break;
            case THREADIDX:
              PUSH(threadIdx.x);
              break;
            case BLOCKIDX:
              PUSH(blockIdx.x);
              break;
            case BLOCKDIMX:
              PUSH(blockDim.x);
              break;
            case EXIT:
              cont = 0;
              break;
            case SYNCTHREADS:
              __syncthreads();
              break;
            }
        }
      
      else if(IIS(opcode))
        PUSH(UNBOX(opcode));
    }
  while(cont);
}


__global__ void interpreter_kernel(word* program) {
  interpreter(program);
}

word* append(word* prog, word code)
{
  prog = (word*)realloc((word*)prog, prog[1] + 1);
  prog[prog[1]] = code;
  prog[1]++;
  return prog;
}


word* read()
{
  char *endptr, *line, *pch = NULL;
  word val;

  size_t line_size = 0;
  size_t token_size = 0;

  word* prog = (word*)malloc(sizeof(word) * 2);
  prog[0] = 0; // header
  prog[1] = 2; // size
  for(;;)
    {
      if(getline(&line, &line_size, stdin) == -1)
        {
          break;
        }
      else
        {
          for(pch = strtok(line, " "); pch != NULL; pch = strtok(NULL, " "))
            {
              token_size = strlen(pch);
              if (pch[token_size-1] == '\n')
                  pch[token_size-1] = 0;

              if (strlen(pch) == 0)
                break;

              errno = 0;
              val = (word)strtoll(pch, &endptr, 0);

              if (errno == ERANGE || (errno != 0 && val == 0)) {
                perror("strtol");
                exit(EXIT_FAILURE);
              }

              if ((endptr == pch) || (*endptr != '\0'))
                  goto core;
              prog = append(prog, IBOX(val));
              continue;

            core:
              for (int i = 0; i < (int)(sizeof(CoreWords)/sizeof(Words)); ++i)
                {
                  if (strcmp(pch, CoreWords[i].name) == 0)
                    {
                      prog = append(prog, OBOX(CoreWords[i].code));
                      break;
                    }
                }
            }
        }
    }
  prog = append(prog, OBOX(EXIT));
  return prog;
}

int mymain(int argc, const char* argv[])
{
  word *prog = read();
  
  int deviceCount = 0;
  int N = 1;
  
  HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
  
  size_t psize = prog[1] * sizeof(word);
  
  word* d_prog;
  cudaMalloc(&d_prog, psize);
  
  cudaMemcpy(d_prog, prog, psize, cudaMemcpyHostToDevice);
  
  interpreter_kernel <<< N, N, N * sizeof(word) * 16>>>(d_prog);
  
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  printf("Bye.");
}
