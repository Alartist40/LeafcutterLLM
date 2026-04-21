#ifndef QKERNEL_H
#define QKERNEL_H

#include <stdint.h>

#define QKERNEL_GROUP_SIZE 64

/*
 * q4_gemm: 4-bit weight GEMM.
 * C[M,N] = A[M,K] × dequant(B_packed[N,K/2], scales_B, zeros_B)
 */
void q4_gemm(
    float*          C,
    const float*    A,
    const uint8_t*  B_packed,
    const float*    scales_B,
    const float*    zeros_B,
    int M, int N, int K,
    int group_size
);

/* Batched variant: M = batch * seq_len */
void q4_gemm_batched(
    float*          C,
    const float*    A,
    const uint8_t*  B_packed,
    const float*    scales_B,
    const float*    zeros_B,
    int batch, int seq_len, int N, int K,
    int group_size
);

#endif /* QKERNEL_H */
