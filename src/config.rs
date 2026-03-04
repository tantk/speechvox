use crate::models;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionMode {
    PushToTalk,
    ContinuousStreaming,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model_dir: String,
    pub hotkey: String,
    pub overlay_x: Option<i32>,
    pub overlay_y: Option<i32>,
    #[serde(default = "default_mode")]
    pub mode: TranscriptionMode,
    #[serde(default)]
    pub active_model: Option<String>,
}

fn default_mode() -> TranscriptionMode {
    TranscriptionMode::PushToTalk
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_dir: String::new(),
            hotkey: "F9".to_string(),
            overlay_x: None,
            overlay_y: None,
            mode: default_mode(),
            active_model: None,
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

    /// Resolve the effective model directory.
    /// 1. active_model + .complete marker → models/{id}/
    /// 2. model_dir non-empty + exists → legacy path
    /// 3. None → no model available
    pub fn effective_model_dir(&self) -> Option<PathBuf> {
        if let Some(ref id) = self.active_model {
            if models::is_model_complete(id) {
                return Some(models::model_dir(id));
            }
        }
        let legacy = PathBuf::from(&self.model_dir);
        if !self.model_dir.is_empty() && legacy.exists() {
            return Some(legacy);
        }
        None
    }

    pub fn validate(&self) -> Result<()> {
        if self.effective_model_dir().is_none() {
            warn!("No valid model directory configured");
        }
        Ok(())
    }
}
