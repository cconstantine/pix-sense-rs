fn main() {
    cc::Build::new()
        .cpp(true)
        .file("trt_wrapper.cpp")
        .include("/usr/include/aarch64-linux-gnu")
        .include("/usr/local/cuda/include")
        .flag("-std=c++17")
        .compile("trt_wrapper");

    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=nvonnxparser");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-search=/usr/lib/aarch64-linux-gnu");
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
    println!("cargo:rerun-if-changed=trt_wrapper.cpp");
    println!("cargo:rerun-if-changed=trt_wrapper.h");
}
