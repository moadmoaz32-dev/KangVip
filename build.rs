fn main() {
    println!("cargo:rerun-if-changed=c_src/core_simd.cpp");

    cc::Build::new()
        .cpp(true)
        .file("c_src/core_simd.cpp")
        .flag("-O3")
        .flag("-mavx2")
        .flag("-mbmi2") // Kích hoạt lệnh nhân số cực lớn của Intel
        .flag("-funroll-loops") // Ép bung vòng lặp
        .compile("core_simd");
}
