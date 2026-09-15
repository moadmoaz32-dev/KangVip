#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

extern "C" {
    // 1. Định nghĩa cấu trúc Struct of Arrays (SoA) tương thích chuẩn C-ABI
    struct U256_SoA {
        uint64_t* l0;
        uint64_t* l1;
        uint64_t* l2;
        uint64_t* l3;
    };

    // Hàm nội tuyến tính cờ nhớ (Carry) bằng SIMD Trick (Đảo bit dấu)
    static inline __m256i get_carry(__m256i a, __m256i sum) {
        __m256i sign_mask = _mm256_set1_epi64x(0x8000000000000000ULL);
        __m256i a_flip = _mm256_xor_si256(a, sign_mask);
        __m256i sum_flip = _mm256_xor_si256(sum, sign_mask);
        __m256i cmp = _mm256_cmpgt_epi64(a_flip, sum_flip);
        return _mm256_srli_epi64(cmp, 63);
    }

    // 2. Hạt nhân Cộng Module P (P của secp256k1)
    void simd_add_mod_p_soa(const U256_SoA* a, const U256_SoA* b, U256_SoA* res, size_t batch_size) {
        // Định nghĩa các Limb của hằng số P = 2^256 - 2^32 - 977
        __m256i p0 = _mm256_set1_epi64x(0xFFFFFFFEFFFFFC2FULL);
        __m256i p1 = _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFFULL);
        __m256i p2 = _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFFULL);
        __m256i p3 = _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFFULL);

        for (size_t i = 0; i < batch_size; i += 4) {
            // --- BƯỚC 1: CỘNG THÔNG THƯỜNG (S = A + B) ---
            __m256i a0 = _mm256_loadu_si256((const __m256i*)&a->l0[i]);
            __m256i b0 = _mm256_loadu_si256((const __m256i*)&b->l0[i]);
            __m256i s0 = _mm256_add_epi64(a0, b0);
            __m256i c0 = get_carry(a0, s0);

            __m256i a1 = _mm256_loadu_si256((const __m256i*)&a->l1[i]);
            __m256i b1 = _mm256_loadu_si256((const __m256i*)&b->l1[i]);
            __m256i s1_tmp = _mm256_add_epi64(a1, b1);
            __m256i s1 = _mm256_add_epi64(s1_tmp, c0);
            __m256i c1 = _mm256_or_si256(get_carry(a1, s1_tmp), get_carry(s1_tmp, s1));

            __m256i a2 = _mm256_loadu_si256((const __m256i*)&a->l2[i]);
            __m256i b2 = _mm256_loadu_si256((const __m256i*)&b->l2[i]);
            __m256i s2_tmp = _mm256_add_epi64(a2, b2);
            __m256i s2 = _mm256_add_epi64(s2_tmp, c1);
            __m256i c2 = _mm256_or_si256(get_carry(a2, s2_tmp), get_carry(s2_tmp, s2));

            __m256i a3 = _mm256_loadu_si256((const __m256i*)&a->l3[i]);
            __m256i b3 = _mm256_loadu_si256((const __m256i*)&b->l3[i]);
            __m256i s3_tmp = _mm256_add_epi64(a3, b3);
            __m256i s3 = _mm256_add_epi64(s3_tmp, c2);
            __m256i c_overflow = _mm256_or_si256(get_carry(a3, s3_tmp), get_carry(s3_tmp, s3));

            // --- BƯỚC 2: RÚT GỌN MODULE (Nếu tràn hoặc S >= P thì S = S - P) ---
            // (Phần này sẽ xử lý trừ đi P bằng SIMD Masked Subtraction)
            // Tạm thời ghi kết quả cơ bản ra mảng:
            _mm256_storeu_si256((__m256i*)&res->l0[i], s0);
            _mm256_storeu_si256((__m256i*)&res->l1[i], s1);
            _mm256_storeu_si256((__m256i*)&res->l2[i], s2);
            _mm256_storeu_si256((__m256i*)&res->l3[i], s3);
        }
    }
}
