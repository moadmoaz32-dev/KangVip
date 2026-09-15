#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

extern "C" {

    // Cấu trúc SoA mapping trực tiếp 1-1 với Rust
    struct U256_SoA {
        uint64_t* l0;
        uint64_t* l1;
        uint64_t* l2;
        uint64_t* l3;
    };

    // Hàm nội tuyến (inline) tính cờ nhớ (Carry) bằng SIMD Bitwise Trick
    static inline __m256i get_carry(__m256i a, __m256i sum) {
        __m256i sign_mask = _mm256_set1_epi64x(0x8000000000000000ULL);
        // Đảo bit cao nhất để chuyển so sánh không dấu (unsigned) thành có dấu (signed)
        __m256i a_flip = _mm256_xor_si256(a, sign_mask);
        __m256i sum_flip = _mm256_xor_si256(sum, sign_mask);
        // Nếu a > sum (nghĩa là đã xảy ra tràn số) -> Trả về cờ nhớ
        __m256i cmp = _mm256_cmpgt_epi64(a_flip, sum_flip);
        return _mm256_srli_epi64(cmp, 63); // Ép về giá trị 1 hoặc 0
    }

    // Hàm cộng song song hàng ngàn số 256-bit siêu tốc
    void simd_add_256_soa(const U256_SoA* a, const U256_SoA* b, U256_SoA* res, size_t batch_size) {
        // Nhảy 4 phần tử mỗi bước (Thanh ghi 256-bit chứa được 4 biến 64-bit)
        for (size_t i = 0; i < batch_size; i += 4) {
            
            // --- LIMB 0 ---
            __m256i a0 = _mm256_loadu_si256((const __m256i*)&a->l0[i]);
            __m256i b0 = _mm256_loadu_si256((const __m256i*)&b->l0[i]);
            __m256i s0 = _mm256_add_epi64(a0, b0);
            __m256i c0 = get_carry(a0, s0); // Lấy cờ nhớ truyền lên limb1
            _mm256_storeu_si256((__m256i*)&res->l0[i], s0);

            // --- LIMB 1 ---
            __m256i a1 = _mm256_loadu_si256((const __m256i*)&a->l1[i]);
            __m256i b1 = _mm256_loadu_si256((const __m256i*)&b->l1[i]);
            __m256i s1_tmp = _mm256_add_epi64(a1, b1);
            __m256i c1_tmp = get_carry(a1, s1_tmp);
            __m256i s1 = _mm256_add_epi64(s1_tmp, c0); // Cộng thêm cờ nhớ từ limb0
            __m256i c1_out = _mm256_or_si256(c1_tmp, get_carry(s1_tmp, s1));
            _mm256_storeu_si256((__m256i*)&res->l1[i], s1);

            // --- LIMB 2 ---
            __m256i a2 = _mm256_loadu_si256((const __m256i*)&a->l2[i]);
            __m256i b2 = _mm256_loadu_si256((const __m256i*)&b->l2[i]);
            __m256i s2_tmp = _mm256_add_epi64(a2, b2);
            __m256i c2_tmp = get_carry(a2, s2_tmp);
            __m256i s2 = _mm256_add_epi64(s2_tmp, c1_out);
            __m256i c2_out = _mm256_or_si256(c2_tmp, get_carry(s2_tmp, s2));
            _mm256_storeu_si256((__m256i*)&res->l2[i], s2);

            // --- LIMB 3 (Không cần lấy cờ nhớ đầu ra vì 256-bit là lớn nhất) ---
            __m256i a3 = _mm256_loadu_si256((const __m256i*)&a->l3[i]);
            __m256i b3 = _mm256_loadu_si256((const __m256i*)&b->l3[i]);
            __m256i s3_tmp = _mm256_add_epi64(a3, b3);
            __m256i s3 = _mm256_add_epi64(s3_tmp, c2_out);
            _mm256_storeu_si256((__m256i*)&res->l3[i], s3);
        }
    }
}