use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
pub struct ModelManifest {
    pub schema_version: u32,
    pub hf_repo: String,
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    pub quantization: String,
    pub vram_mb: u64,
    #[serde(default)]
    pub recommended: bool,
    pub files: Vec<ModelFile>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelFile {
    pub name: String,
    pub hf_path: String,
    pub size_bytes: u64,
}

impl ModelEntry {
    pub fn total_download_size(&self) -> u64 {
        self.files.iter().map(|f| f.size_bytes).sum()
    }
}

const MANIFEST_JSON: &str = include_str!("../models.json");

pub fn load_manifest() -> ModelManifest {
    serde_json::from_str(MANIFEST_JSON).expect("bundled models.json is invalid")
}

pub fn models_dir() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    exe_dir.join("models")
}

pub fn model_dir(id: &str) -> PathBuf {
    models_dir().join(id)
}

pub fn is_model_complete(id: &str) -> bool {
    let dir = model_dir(id);
    // Fast path: explicit marker from a completed download
    if dir.join(".complete").exists() {
        return true;
    }
    // Slow path: check if all files already exist with correct sizes
    // (e.g. user manually placed them or copied from another machine)
    let manifest = load_manifest();
    if let Some(model) = manifest.models.iter().find(|m| m.id == id) {
        let all_present = model.files.iter().all(|f| {
            let path = dir.join(&f.name);
            match std::fs::metadata(&path) {
                Ok(meta) => meta.len() == f.size_bytes,
                Err(_) => false,
            }
        });
        if all_present {
            // Write the marker so future checks are fast
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(dir.join(".complete"), "");
            return true;
        }
    }
    false
}

pub fn list_local_models() -> Vec<String> {
    let dir = models_dir();
    if !dir.exists() {
        return Vec::new();
    }
    let mut result = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    if entry.path().join(".complete").exists() {
                        result.push(name.to_string());
                    }
                }
            }
        }
    }
    result
}
