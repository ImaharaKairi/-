**矩阵乘法优化**
本项目将记录通用矩阵乘法在 CPU 上的循序渐进的优化过程。

**测试环境与 Benchmark 指标**
CPU: Intel Core Ultra 9 275HX
OS: Windows 11
Compiler: MinGW-w64 GCC (Dev-C++)
矩阵大小 (N): 1024 x 1024 (单精度浮点数)
性能指标: GFLOPS (每秒十亿次浮点运算)
计算公式: N*N的矩阵乘法需要N^3次乘法和N^3次加法，总运算次数为2*N^3。
GFLOPS = (2.0 * N * N * N) / (time(s) * 1e9)

**VO_i-j-k**
严格按照矩阵乘法数学定义的代码:

void sgemm_v0(int N, float* A, float* B, float* C) {
    for (int i = 0; i < N; ++i) {           // 遍历 C 的行
        for (int j = 0; j < N; ++j) {       // 遍历 C 的列
            float sum = 0.0f;               
            for (int k = 0; k < N; ++k) {   // 计算点积
                // A 是按行读取 (连续访问)
                // B 是按列读取 (跳跃访问，步长为 N)
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;             
        }
    }
}

**V1_i-k-j**
交换j循环与k循环的顺序：

void sgemm_v1(int N, float* A, float* B, float* C) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            // 优化1：寄存器驻留。内层循环中 i 和 k 是常数。
            float a_ik = A[i * N + k]; 
            
            for (int j = 0; j < N; ++j) {
                // 优化2：B 和 C 均变为严格的连续访问
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

**V2_增加循环步长**
在最内层循环，手动将单次处理的元素增加到 4 个，并将循环步长修改为 4：

void sgemm_v2(int N, float* A, float* B, float* C) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float a_ik = A[i * N + k];
            // 步长修改为 4
            for (int j = 0; j < N; j += 4) {
                // 四次乘加运算相互独立
                C[i * N + j]     += a_ik * B[k * N + j];
                C[i * N + j + 1] += a_ik * B[k * N + j + 1];
                C[i * N + j + 2] += a_ik * B[k * N + j + 2];
                C[i * N + j + 3] += a_ik * B[k * N + j + 3];
            }
        }
    }
}

**V3_**
使用 AVX2 函数，smid向量化，一次处理八个浮点数：

#include <immintrin.h> // 引入 SIMD 头文件

void sgemm_v3_avx2(int N, float* A, float* B, float* C) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            // _mm256_set1_ps：将 1 个标量广播到 256 位寄存器的 8 个槽中
            __m256 a_ik_vec = _mm256_set1_ps(A[i * N + k]); 
            
            // 步长修改为 8 (256 bits = 8 x 32 bits)
            for (int j = 0; j < N; j += 8) {
                // _mm256_loadu_ps：一条指令从内存连续读取 8 个 float
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]); 
                __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]); 
                
                // _mm256_fmadd_ps：核心算力来源，乘加融合指令
                c_vec = _mm256_fmadd_ps(a_ik_vec, b_vec, c_vec); 
                
                // _mm256_storeu_ps：一条指令将 8 个结果写回内存
                _mm256_storeu_ps(&C[i * N + j], c_vec); 
            }
        }
    }
}

V4
将 1024*1024的大矩阵，在逻辑上划分成多个64*64的小块进行计算。

#define BLOCK_SIZE 64 

void sgemm_v4_tiling(int N, float* A, float* B, float* C) {
    // 外部循环：遍历矩阵宏块 (Macro-blocks)
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int k = 0; k < N; k += BLOCK_SIZE) {
            for (int j = 0; j < N; j += BLOCK_SIZE) {
                
                // 内部循环：在 64x64 的 Micro-block 内部执行 AVX2 计算
                for (int ii = i; ii < i + BLOCK_SIZE; ++ii) {
                    for (int kk = k; kk < k + BLOCK_SIZE; ++kk) {
                        __m256 a_vec = _mm256_set1_ps(A[ii * N + kk]);
                        for (int jj = j; jj < j + BLOCK_SIZE; jj += 8) {
                            __m256 b_vec = _mm256_loadu_ps(&B[kk * N + jj]);
                            __m256 c_vec = _mm256_loadu_ps(&C[ii * N + jj]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(&C[ii * N + jj], c_vec);
                        }
                    }
                }
                
            }
        }
    }
}

V5
利用 OpenMP 编译器指令，在线程级别物理拆分任务：

#include <omp.h>

void sgemm_v5_openmp(int N, float* A, float* B, float* C) {
    // 编译制导指令：
    // parallel for: 创建线程组并分发 for 循环迭代
    // collapse(2): 将外层 2 维循环展平为一维循环空间，提供充足的细粒度任务
    // schedule(dynamic): 动态负载均衡调度
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            // 对独立的 C 矩阵子块进行累加运算
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                for (int ii = i; ii < i + BLOCK_SIZE; ++ii) {
                    for (int kk = k; kk < k + BLOCK_SIZE; ++kk) {
                        __m256 a_vec = _mm256_set1_ps(A[ii * N + kk]);
                        for (int jj = j; jj < j + BLOCK_SIZE; jj += 8) {
                            __m256 b_vec = _mm256_loadu_ps(&B[kk * N + jj]);
                            __m256 c_vec = _mm256_loadu_ps(&C[ii * N + jj]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(&C[ii * N + jj], c_vec);
                        }
                    }
                }
            }
        }
    }
}
