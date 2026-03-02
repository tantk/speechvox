use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model_path: String,
    pub mmproj_path: String,
    pub hotkey: String,
    pub use_gpu: bool,
    pub overlay_x: Option<i32>,
    pub overlay_y: Option<i32>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: "C:\\dev\\mistralhx\\Voxtral-Mini-4B-Realtime-2602-GGUF_Q4_K_M.gguf"
                .to_string(),
            mmproj_path: "C:\\dev\\mistralhx\\voxtral-realtime-4b-mmproj-f16.gguf".to_string(),
            hotkey: "F9".to_string(),
            use_gpu: true,
            overlay_x: None,
            overlay_y: None,
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path();

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let config: Config = serde_json::from_str(&content)?;
            info!("Config loaded from {}", config_path.display());
            Ok(config)
        } else {
            let config = Config::default();
            config.save()?;
            info!("Default config created at {}", config_path.display());
            Ok(config)
        }
    }

    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path();
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&config_path, content)?;
        Ok(())
    }

    fn config_path() -> PathBuf {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."));
        exe_dir.join("speechvox.json")
    }

    pub fn validate(&self) -> Result<()> {
        if !PathBuf::from(&self.model_path).exists() {
            warn!("Model file not found: {}", self.model_path);
        }
        if !PathBuf::from(&self.mmproj_path).exists() {
            warn!("MMProj file not found: {}", self.mmproj_path);
        }
        Ok(())
    }
}
