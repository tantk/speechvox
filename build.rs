use std::path::PathBuf;

fn main() {
    // Force CMake to use Visual Studio 2022 (not the year from system date)
    // This fixes the "Visual Studio 18 2026" error on systems with dates >= 2026
    if std::env::var("CMAKE_GENERATOR").is_err() {
        std::env::set_var("CMAKE_GENERATOR", "Visual Studio 17 2022");
    }

    let llama_cpp_dir = PathBuf::from("C:/dev/mistralhx/llama.cpp");
    if !llama_cpp_dir.exists() {
        panic!("llama.cpp source not found at {}", llama_cpp_dir.display());
    }

    // Force CUDA 13.0 (avoid stale CUDA 10.1 headers on PATH)
    let cuda_path = std::env::var("CUDA_PATH").unwrap_or_else(|_| {
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0".to_string()
    });
    let cuda_compiler = PathBuf::from(&cuda_path).join("bin").join("nvcc.exe");

    let dst = cmake::Config::new(&llama_cpp_dir)
        .define("GGML_CUDA", "ON")
        .define("GGML_STATIC", "ON")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_TOOLS", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release")
        // CUDA 13.0 minimum is compute_75; target native arch (RTX 4070 Ti = sm_89)
        .define("CMAKE_CUDA_ARCHITECTURES", "89")
        // Pin to CUDA 13.0 to avoid picking up old CUDA 10.1 headers
        .define("CMAKE_CUDA_COMPILER", cuda_compiler.to_str().unwrap())
        .define("CUDAToolkit_ROOT", &cuda_path)
        .build_target("mtmd")
        .build();

    let build_dir = dst.join("build");

    // Tell cargo where to find the static libraries
    // MSVC multi-config generators put libs in Debug/ or Release/ subdirs
    let lib_dirs = [
        // llama.lib
        build_dir.join("src").join("Release"),
        build_dir.join("src").join("Debug"),
        // ggml.lib, ggml-base.lib, ggml-cpu.lib
        build_dir.join("ggml").join("src").join("Release"),
        build_dir.join("ggml").join("src").join("Debug"),
        // ggml-cuda.lib
        build_dir.join("ggml").join("src").join("ggml-cuda").join("Release"),
        build_dir.join("ggml").join("src").join("ggml-cuda").join("Debug"),
        // mtmd.lib
        build_dir.join("tools").join("mtmd").join("Release"),
        build_dir.join("tools").join("mtmd").join("Debug"),
    ];

    for dir in &lib_dirs {
        if dir.exists() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }

    // Link llama.cpp static libraries
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-cuda");
    println!("cargo:rustc-link-lib=static=mtmd");

    // Link CUDA runtime libraries
    let cuda_lib_dir = PathBuf::from(&cuda_path).join("lib").join("x64");
    if cuda_lib_dir.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    }

    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");
    println!("cargo:rustc-link-lib=cuda"); // CUDA driver API

    // Link C++ standard library from the VS 2022 toolset that cmake used
    // (Rust's default linker uses MSVC 14.29 which lacks newer STL vectorized symbols)
    let vs2022_lib = PathBuf::from(
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/lib/x64",
    );
    if vs2022_lib.exists() {
        println!("cargo:rustc-link-search=native={}", vs2022_lib.display());
    }
    println!("cargo:rustc-link-lib=msvcprt"); // C++ STL

    // Link Windows system libraries
    println!("cargo:rustc-link-lib=user32");
    println!("cargo:rustc-link-lib=shell32");
    println!("cargo:rustc-link-lib=advapi32");
    println!("cargo:rustc-link-lib=ole32");

    // Only rebuild if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
}

