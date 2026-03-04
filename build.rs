use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Vendored voxtral.c source (relative to Cargo.toml)
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let voxtral_dir = manifest_dir.join("vendor").join("voxtral");
    if !voxtral_dir.exists() {
        panic!(
            "voxtral.c source not found at {}",
            voxtral_dir.display()
        );
    }

    // CUDA toolkit path
    let cuda_path = std::env::var("CUDA_PATH").unwrap_or_else(|_| {
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0".to_string()
    });
    let cuda_include = PathBuf::from(&cuda_path).join("include");

    // --- Fatbin multi-arch CUDA compilation ---
    let cuda_arch_label = compile_cuda_kernels(&voxtral_dir, &cuda_path, &cuda_include);

    // Source files (CUDA target — excludes main.c, mic files, cuda_stub, inspect_weights)
    let sources = [
        "voxtral.c",
        "voxtral_kernels.c",
        "voxtral_audio.c",
        "voxtral_encoder.c",
        "voxtral_decoder.c",
        "voxtral_tokenizer.c",
        "voxtral_safetensors.c",
        "voxtral_quant_loader.c",
        "voxtral_quant_kernels.c",
        "voxtral_cuda.c",
        "voxtral_cuda_quant.c",
    ];

    let mut build = cc::Build::new();
    build
        .warnings(false)
        .define("USE_CUDA", None)
        .define("VOX_CUDA_ARCH", cuda_arch_label.as_str())
        .include(&voxtral_dir)
        .include(&cuda_include)
        .flag_if_supported("/O2") // MSVC optimization
        .flag_if_supported("-O3"); // GCC/clang optimization

    for src in &sources {
        build.file(voxtral_dir.join(src));
    }

    build.compile("voxtral");

    // Link CUDA libraries
    let cuda_lib_dir = PathBuf::from(&cuda_path).join("lib").join("x64");
    if cuda_lib_dir.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    }

    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");
    println!("cargo:rustc-link-lib=cuda");

    // Link C++ standard library (needed by CUDA runtime)
    let vs2022_lib = PathBuf::from(
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/lib/x64",
    );
    if vs2022_lib.exists() {
        println!("cargo:rustc-link-search=native={}", vs2022_lib.display());
    }
    println!("cargo:rustc-link-lib=msvcprt");

    // Windows system libraries
    println!("cargo:rustc-link-lib=user32");
    println!("cargo:rustc-link-lib=shell32");
    println!("cargo:rustc-link-lib=advapi32");
    println!("cargo:rustc-link-lib=ole32");

    // Embed app icon into the Windows executable
    let mut res = winresource::WindowsResource::new();
    res.set_icon("assets/icon.ico");
    res.compile().expect("Failed to compile Windows resources");

    // Rerun if sources change
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/icon.ico");
    println!(
        "cargo:rerun-if-changed={}",
        voxtral_dir.join("voxtral_cuda_kernels.cu").display()
    );
    for src in &sources {
        println!(
            "cargo:rerun-if-changed={}",
            voxtral_dir.join(src).display()
        );
    }
    println!("cargo:rerun-if-env-changed=SPEECHVOX_NVCC");
    println!("cargo:rerun-if-env-changed=SPEECHVOX_CUDA_ARCH");
}

/// Compile CUDA kernels into a fatbin (or single-arch cubin) and generate
/// the C header. Returns the arch label for VOX_CUDA_ARCH define.
fn compile_cuda_kernels(voxtral_dir: &PathBuf, cuda_path: &str, cuda_include: &PathBuf) -> String {
    let cu_source = voxtral_dir.join("voxtral_cuda_kernels.cu");
    let header_path = voxtral_dir.join("voxtral_cuda_kernels_cubin.h");

    // Find nvcc
    let nvcc = find_nvcc(cuda_path);
    let nvcc = match nvcc {
        Some(p) => p,
        None => {
            println!("cargo:warning=nvcc not found — using pre-built cubin header as fallback");
            return "sm_89".to_string();
        }
    };

    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Find cl.exe for nvcc (it needs MSVC host compiler in PATH)
    let cl_path = find_cl_exe();

    // Check for single-arch dev override
    if let Ok(arch) = std::env::var("SPEECHVOX_CUDA_ARCH") {
        // Single-arch cubin (fast dev builds)
        let cubin_path = PathBuf::from(&out_dir).join("voxtral_cuda_kernels.cubin");
        let mut cmd = Command::new(&nvcc);
        if let Some(ref cl_dir) = cl_path {
            prepend_path(&mut cmd, cl_dir);
        }
        let status = cmd
            .arg("--cubin")
            .arg("-O3")
            .arg("--std=c++14")
            .arg("-lineinfo")
            .arg(format!("-arch={}", arch))
            .arg(format!("-I{}", cuda_include.display()))
            .arg("-o")
            .arg(&cubin_path)
            .arg(&cu_source)
            .status();

        match status {
            Ok(s) if s.success() => {
                generate_cubin_header(&cubin_path, &header_path);
                return arch;
            }
            Ok(s) => {
                println!(
                    "cargo:warning=nvcc single-arch failed (exit {}), using pre-built header",
                    s
                );
                return arch;
            }
            Err(e) => {
                println!("cargo:warning=nvcc execution failed: {}, using pre-built header", e);
                return arch;
            }
        }
    }

    // Full fatbin: detect supported architectures
    let mut gencode_args = vec![
        "-gencode".to_string(), "arch=compute_75,code=sm_75".to_string(),
        "-gencode".to_string(), "arch=compute_86,code=sm_86".to_string(),
        "-gencode".to_string(), "arch=compute_89,code=sm_89".to_string(),
    ];

    // Check if sm_100 is supported (CUDA >= 12.8)
    if nvcc_supports_arch(&nvcc, "sm_100") {
        gencode_args.push("-gencode".to_string());
        gencode_args.push("arch=compute_100,code=sm_100".to_string());
    }

    // PTX forward-compat fallback (JIT for future GPUs)
    gencode_args.push("-gencode".to_string());
    gencode_args.push("arch=compute_75,code=compute_75".to_string());

    let fatbin_path = PathBuf::from(&out_dir).join("voxtral_cuda_kernels.fatbin");
    let mut cmd = Command::new(&nvcc);
    if let Some(ref cl_dir) = cl_path {
        prepend_path(&mut cmd, cl_dir);
    }
    let status = cmd
        .arg("--fatbin")
        .arg("-O3")
        .arg("--std=c++14")
        .arg("-lineinfo")
        .args(&gencode_args)
        .arg(format!("-I{}", cuda_include.display()))
        .arg("-o")
        .arg(&fatbin_path)
        .arg(&cu_source)
        .status();

    match status {
        Ok(s) if s.success() => {
            generate_cubin_header(&fatbin_path, &header_path);
            "fatbin".to_string()
        }
        Ok(s) => {
            println!(
                "cargo:warning=nvcc fatbin failed (exit {}), using pre-built header",
                s
            );
            "sm_89".to_string()
        }
        Err(e) => {
            println!("cargo:warning=nvcc execution failed: {}, using pre-built header", e);
            "sm_89".to_string()
        }
    }
}

/// Find nvcc: check SPEECHVOX_NVCC env, then {CUDA_PATH}/bin/nvcc.exe
fn find_nvcc(cuda_path: &str) -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SPEECHVOX_NVCC") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
        println!("cargo:warning=SPEECHVOX_NVCC set to '{}' but file not found", path);
    }

    let nvcc = PathBuf::from(cuda_path).join("bin").join("nvcc.exe");
    if nvcc.exists() {
        return Some(nvcc);
    }

    None
}

/// Check if nvcc supports a given GPU architecture (e.g. "sm_100").
/// `nvcc --list-gpu-arch` outputs `compute_XX` lines.
fn nvcc_supports_arch(nvcc: &PathBuf, arch: &str) -> bool {
    let output = Command::new(nvcc)
        .arg("--list-gpu-arch")
        .output();

    // Convert "sm_100" to "compute_100" for matching
    let compute_arch = arch.replace("sm_", "compute_");

    match output {
        Ok(o) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            stdout.lines().any(|line| line.trim() == compute_arch)
        }
        _ => false,
    }
}

/// Find cl.exe (MSVC compiler) so nvcc can use it as host compiler.
/// Checks the CC env var first, then known VS 2022 paths.
fn find_cl_exe() -> Option<PathBuf> {
    // Check if cc crate set the compiler path
    if let Ok(cc) = std::env::var("CC") {
        let p = PathBuf::from(&cc);
        if p.exists() {
            return p.parent().map(|p| p.to_path_buf());
        }
    }

    // Known VS 2022 paths
    let candidates = [
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC",
        "C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/MSVC",
        "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/MSVC",
        "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC",
    ];

    for base in &candidates {
        let base = PathBuf::from(base);
        if !base.exists() {
            continue;
        }
        // Find the latest version directory
        if let Ok(entries) = std::fs::read_dir(&base) {
            let mut versions: Vec<_> = entries
                .flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| e.path())
                .collect();
            versions.sort();
            if let Some(latest) = versions.last() {
                let cl = latest.join("bin/Hostx64/x64/cl.exe");
                if cl.exists() {
                    return cl.parent().map(|p| p.to_path_buf());
                }
            }
        }
    }

    None
}

/// Prepend a directory to the PATH env var for a Command
fn prepend_path(cmd: &mut Command, dir: &PathBuf) {
    let current = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{};{}", dir.display(), current);
    cmd.env("PATH", new_path);
}

/// Read binary file and generate a C header with the same variable names
/// as the original xxd-generated header (voxtral_cuda_kernels_cubin[]).
fn generate_cubin_header(binary_path: &PathBuf, header_path: &PathBuf) {
    let data = std::fs::read(binary_path).unwrap_or_else(|e| {
        panic!("Failed to read {}: {}", binary_path.display(), e);
    });

    let mut header = String::with_capacity(data.len() * 6 + 256);
    header.push_str("unsigned char voxtral_cuda_kernels_cubin[] = {\n");

    for (i, byte) in data.iter().enumerate() {
        if i % 12 == 0 {
            header.push_str("  ");
        }
        header.push_str(&format!("0x{:02x}", byte));
        if i < data.len() - 1 {
            header.push_str(", ");
        }
        if i % 12 == 11 {
            header.push('\n');
        }
    }

    header.push_str("\n};\n");
    header.push_str(&format!(
        "unsigned int voxtral_cuda_kernels_cubin_len = {};\n",
        data.len()
    ));

    std::fs::write(header_path, header).unwrap_or_else(|e| {
        panic!("Failed to write {}: {}", header_path.display(), e);
    });

    println!(
        "cargo:warning=Generated CUDA kernel header: {} ({} bytes)",
        header_path.display(),
        data.len()
    );
}
