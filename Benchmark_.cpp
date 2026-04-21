#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <string>
#include <immintrin.h> // SIMD 指令头文件
#include <omp.h>       // OpenMP 多线程头文件

// 定义矩阵大小和分块大小
const int N = 1024;
#define BLOCK_SIZE 64

// V0
void sgemm_v0(int n, float* A, float* B, float* C) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    }
}

// V1
void sgemm_v1(int n, float* A, float* B, float* C) {
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            float a_ik = A[i * n + k];
            for (int j = 0; j < n; ++j) C[i * n + j] += a_ik * B[k * n + j];
        }
    }
}

// V2
void sgemm_v2(int n, float* A, float* B, float* C) {
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            float a_ik = A[i * n + k];
            for (int j = 0; j < n; j += 4) {
                C[i * n + j]     += a_ik * B[k * n + j];
                C[i * n + j + 1] += a_ik * B[k * n + j + 1];
                C[i * n + j + 2] += a_ik * B[k * n + j + 2];
                C[i * n + j + 3] += a_ik * B[k * n + j + 3];
            }
        }
    }
}

// V3: AVX2 向量化
void sgemm_v3_avx2(int n, float* A, float* B, float* C) {
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            __m256 a_ik_vec = _mm256_set1_ps(A[i * n + k]);
            for (int j = 0; j < n; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(&B[k * n + j]);
                __m256 c_vec = _mm256_loadu_ps(&C[i * n + j]);
                c_vec = _mm256_fmadd_ps(a_ik_vec, b_vec, c_vec);
                _mm256_storeu_ps(&C[i * n + j], c_vec);
            }
        }
    }
}

// V4 Cache Tiling 
void sgemm_v4_tiling(int n, float* A, float* B, float* C) {
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int k = 0; k < n; k += BLOCK_SIZE) {
            for (int j = 0; j < n; j += BLOCK_SIZE) {
                for (int ii = i; ii < i + BLOCK_SIZE; ++ii) {
                    for (int kk = k; kk < k + BLOCK_SIZE; ++kk) {
                        __m256 a_vec = _mm256_set1_ps(A[ii * n + kk]);
                        for (int jj = j; jj < j + BLOCK_SIZE; jj += 8) {
                            __m256 b_vec = _mm256_loadu_ps(&B[kk * n + jj]);
                            __m256 c_vec = _mm256_loadu_ps(&C[ii * n + jj]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(&C[ii * n + jj], c_vec);
                        }
                    }
                }
            }
        }
    }
}

// V5 openmp
void sgemm_v5_openmp(int n, float* A, float* B, float* C) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int k = 0; k < n; k += BLOCK_SIZE) {
                for (int ii = i; ii < i + BLOCK_SIZE; ++ii) {
                    for (int kk = k; kk < k + BLOCK_SIZE; ++kk) {
                        __m256 a_vec = _mm256_set1_ps(A[ii * n + kk]);
                        for (int jj = j; jj < j + BLOCK_SIZE; jj += 8) {
                            __m256 b_vec = _mm256_loadu_ps(&B[kk * n + jj]);
                            __m256 c_vec = _mm256_loadu_ps(&C[ii * n + jj]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(&C[ii * n + jj], c_vec);
                        }
                    }
                }
            }
        }
    }
}

// 跑分 
void benchmark_func(const std::string& name, void (*func)(int, float*, float*, float*), int n, float* A, float* B, float* C) {
    // 每次测试前，将矩阵 C 清零
    for(int i = 0; i < n * n; ++i) C[i] = 0.0f;
    
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    // 执行测试函数
    func(n, A, B, C);
    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    double seconds = diff.count();
    double gflops = (2.0 * n * n * n) / (seconds * 1e9);
    
    std::cout << "[" << name << "]\t耗时: " << seconds << " 秒 \t性能: " << gflops << " GFLOPS" << std::endl;
}

int main() {
    std::cout << "初始化矩阵 (N=" << N << ")" << std::endl;
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N, 0.0f);

    // 随机填充数据
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    std::cout << "跑分：" << std::endl;
    
    // 依次进行性能评测
    benchmark_func("V0: ", sgemm_v0, N, A.data(), B.data(), C.data());
    benchmark_func("V1: ", sgemm_v1, N, A.data(), B.data(), C.data());
    benchmark_func("V2: ", sgemm_v2, N, A.data(), B.data(), C.data());
    benchmark_func("V3: ", sgemm_v3_avx2, N, A.data(), B.data(), C.data());
    benchmark_func("V4: ", sgemm_v4_tiling, N, A.data(), B.data(), C.data());
    benchmark_func("V5: ", sgemm_v5_openmp, N, A.data(), B.data(), C.data());
    
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "测试完成，按enter退出" << std::endl;
    std::cin.get();
    return 0;
}

