fn main() {
    // Chỉ biên dịch lại C++ nếu file này bị sửa đổi
    println!("cargo:rerun-if-changed=c_src/core_simd.cpp");

    cc::Build::new()
        .cpp(true)
        .file("c_src/core_simd.cpp")
        .flag("-mavx2")     // Kích hoạt thanh ghi 256-bit
        .flag("-O3")        // Tối ưu hóa cực hạn
        .flag("-std=c++17") // Dùng chuẩn C++17
        .compile("core_simd");
}