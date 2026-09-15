#include <stdint.h>
#include <stddef.h>

extern "C" {
    // Hằng số P của secp256k1
    constexpr uint64_t P0 = 0xFFFFFFFEFFFFFC2FULL;
    constexpr uint64_t P1 = 0xFFFFFFFFFFFFFFFFULL;
    constexpr uint64_t P2 = 0xFFFFFFFFFFFFFFFFULL;
    constexpr uint64_t P3 = 0xFFFFFFFFFFFFFFFFULL;
    
    // Hằng số Montgomery: MU = -P^-1 mod 2^64
    constexpr uint64_t MU = 0xD838091DD2253531ULL;

    // Nhân 2 số 256-bit và rút gọn Montgomery siêu tốc
    inline void mont_mul(const uint64_t* a, const uint64_t* b, uint64_t* res) {
        uint64_t t[5] = {0};
        for (int i = 0; i < 4; i++) {
            uint64_t carry = 0;
            for (int j = 0; j < 4; j++) {
                // Sử dụng __uint128_t để phần cứng tự động gọi lệnh MULX
                __uint128_t prod = (__uint128_t)a[i] * b[j] + t[j] + carry;
                t[j] = (uint64_t)prod;
                carry = (uint64_t)(prod >> 64);
            }
            t[4] = carry;

            uint64_t m = t[0] * MU;
            carry = 0;
            __uint128_t prod = (__uint128_t)m * P0 + t[0] + carry;
            carry = (uint64_t)(prod >> 64);

            prod = (__uint128_t)m * P1 + t[1] + carry;
            t[0] = (uint64_t)prod; carry = (uint64_t)(prod >> 64);

            prod = (__uint128_t)m * P2 + t[2] + carry;
            t[1] = (uint64_t)prod; carry = (uint64_t)(prod >> 64);

            prod = (__uint128_t)m * P3 + t[3] + carry;
            t[2] = (uint64_t)prod; carry = (uint64_t)(prod >> 64);

            t[3] = t[4] + carry;
        }

        // Trừ đi P nếu kết quả tràn
        uint64_t sub[4];
        uint64_t borrow = 0;
        __int128_t diff;
        diff = (__int128_t)t[0] - P0 - borrow; sub[0] = (uint64_t)diff; borrow = (diff < 0) ? 1 : 0;
        diff = (__int128_t)t[1] - P1 - borrow; sub[1] = (uint64_t)diff; borrow = (diff < 0) ? 1 : 0;
        diff = (__int128_t)t[2] - P2 - borrow; sub[2] = (uint64_t)diff; borrow = (diff < 0) ? 1 : 0;
        diff = (__int128_t)t[3] - P3 - borrow; sub[3] = (uint64_t)diff; borrow = (diff < 0) ? 1 : 0;

        if (borrow == 0) {
            res[0] = sub[0]; res[1] = sub[1]; res[2] = sub[2]; res[3] = sub[3];
        } else {
            res[0] = t[0]; res[1] = t[1]; res[2] = t[2]; res[3] = t[3];
        }
    }

    // Phép trừ Modulo P (A - B mod P)
    inline void mod_sub(const uint64_t* a, const uint64_t* b, uint64_t* res) {
        uint64_t borrow = 0;
        __int128_t diff;
        diff = (__int128_t)a[0] - b[0] - borrow; res[0] = (uint64_t)diff; borrow = (diff < 0) ? 1 : 0;
        diff = (__int128_t)a[1] - b[1] - borrow; res[1] = (uint64_t)diff; borrow = (diff < 0) ? 1 : 0;
        diff = (__int128_t)a[2] - b[2] - borrow; res[2] = (uint64_t)diff; borrow = (diff < 0) ? 1 : 0;
        diff = (__int128_t)a[3] - b[3] - borrow; res[3] = (uint64_t)diff; borrow = (diff < 0) ? 1 : 0;

        if (borrow) {
            uint64_t carry = 0;
            __uint128_t sum;
            sum = (__uint128_t)res[0] + P0 + carry; res[0] = (uint64_t)sum; carry = (uint64_t)(sum >> 64);
            sum = (__uint128_t)res[1] + P1 + carry; res[1] = (uint64_t)sum; carry = (uint64_t)(sum >> 64);
            sum = (__uint128_t)res[2] + P2 + carry; res[2] = (uint64_t)sum; carry = (uint64_t)(sum >> 64);
            sum = (__uint128_t)res[3] + P3 + carry; res[3] = (uint64_t)sum;
        }
    }

    // MAIN ENDPOINT: Tính toán song song X_new và Y_new cho hàng ngàn điểm
    void c_point_update(
        const uint64_t* px, const uint64_t* py,
        const uint64_t* qx, const uint64_t* qy,
        const uint64_t* inv,
        uint64_t* rx, uint64_t* ry,
        size_t batch_size
    ) {
        // Trình biên dịch sẽ tự động bung luồng tại đây
        for (size_t i = 0; i < batch_size; i++) {
            size_t offset = i * 4;
            
            uint64_t dy[4]; 
            mod_sub(&qy[offset], &py[offset], dy); // Y_Q - Y_P
            
            uint64_t lambda[4];
            mont_mul(dy, &inv[offset], lambda); // lambda = dy * inv
            
            uint64_t lambda_sq[4];
            mont_mul(lambda, lambda, lambda_sq); // lambda^2

            uint64_t rx_tmp[4];
            mod_sub(lambda_sq, &px[offset], rx_tmp); // lambda^2 - X_P
            mod_sub(rx_tmp, &qx[offset], &rx[offset]); // R_X = rx_tmp - X_Q

            uint64_t dx[4];
            mod_sub(&px[offset], &rx[offset], dx); // X_P - R_X
            
            uint64_t ry_tmp[4];
            mont_mul(lambda, dx, ry_tmp); // lambda * dx
            mod_sub(ry_tmp, &py[offset], &ry[offset]); // R_Y = ry_tmp - Y_P
        }
    }
}
