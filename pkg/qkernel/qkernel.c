#include "qkernel.h"
#include <stdlib.h>
#include <string.h>

/*
 * q4_gemm: 4-bit weight × float32 activation matrix multiplication.
 *
 * Layout conventions (matching HuggingFace NF4/Q4_0):
 *   - B is packed: each byte holds TWO 4-bit weights (lo nibble = col 2k, hi nibble = col 2k+1).
 *   - scales_B has one float32 scale per group of QKERNEL_GROUP_SIZE weights in B.
 *   - A is a dense float32 matrix [M × K].
 *   - C is the output float32 matrix [M × N].
 *
 * Safety guarantees:
 *   1. NULL pointer guard at the top.
 *   2. All index calculations use size_t to prevent integer overflow.
 *   3. The packed B byte index is computed as (k >> 1) to avoid OOB reads.
 */
void q4_gemm(
    float*          C,          /* output  [M x N] float32              */
    const float*    A,          /* input   [M x K] float32              */
    const uint8_t*  B_packed,   /* weights [N x K] packed 4-bit         */
    const float*    scales_B,   /* scales  [N x (K/GROUP)] float32      */
    const float*    zeros_B,    /* zero-points [N x (K/GROUP)] float32  */
    int M, int N, int K,
    int group_size
) {
    /* --- Safety: null-guard ---------------------------------------- */
    if (!C || !A || !B_packed || !scales_B) return;
    if (M <= 0 || N <= 0 || K <= 0 || group_size <= 0) return;

    int num_groups = (K + group_size - 1) / group_size;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;

            for (int k = 0; k < K; k++) {
                int group_idx = k / group_size;
                /* scales_B layout: [N, num_groups] */
                float scale = scales_B[(size_t)n * num_groups + group_idx];
                float zero  = zeros_B ? zeros_B[(size_t)n * num_groups + group_idx] : 8.0f;

                /* Unpack 4-bit weight from B_packed.
                 * B_packed layout: [N, K/2] bytes.
                 * byte index = n*(K/2) + k/2
                 * Even k → lo nibble; odd k → hi nibble. */
                size_t byte_idx = (size_t)n * ((K + 1) / 2) + (k >> 1);
                uint8_t packed  = B_packed[byte_idx];
                uint8_t nibble  = (k & 1) ? (packed >> 4) : (packed & 0x0F);

                /* Dequantize in-register: weight = scale * (nibble - zero) */
                float w = scale * ((float)nibble - zero);

                acc += A[(size_t)m * K + k] * w;
            }

            C[(size_t)m * N + n] = acc;
        }
    }
}

/*
 * q4_gemm_batched: Convenience wrapper for processing multiple (M) rows at once.
 * Delegates directly to q4_gemm since q4_gemm already handles batched M.
 */
void q4_gemm_batched(
    float*          C,
    const float*    A,
    const uint8_t*  B_packed,
    const float*    scales_B,
    const float*    zeros_B,
    int batch, int seq_len, int N, int K,
    int group_size
) {
    int M = batch * seq_len;
    q4_gemm(C, A, B_packed, scales_B, zeros_B, M, N, K, group_size);
}
