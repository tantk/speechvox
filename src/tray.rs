use crate::config::TranscriptionMode;
use anyhow::Result;
use image::GenericImageView;
use tray_icon::{
    menu::{Menu, MenuId, MenuItem, PredefinedMenuItem, Submenu},
    Icon, TrayIcon, TrayIconBuilder,
};

const ICON_GRAY: &[u8] = include_bytes!("../assets/icon.png");

pub struct TrayManager {
    _tray: TrayIcon,
    pub exit_id: MenuId,
    pub mode_ptt_id: MenuId,
    pub mode_continuous_id: MenuId,
    pub hotkey_id: MenuId,
    pub models_id: MenuId,
}

impl TrayManager {
    pub fn new(initial_mode: TranscriptionMode) -> Result<Self> {
        let icon = load_png_icon(ICON_GRAY)?;

        // Mode submenu
        let ptt_check = if initial_mode == TranscriptionMode::PushToTalk { " *" } else { "" };
        let cont_check = if initial_mode == TranscriptionMode::ContinuousStreaming { " *" } else { "" };

        let mode_ptt = MenuItem::new(format!("Push-to-Talk{}", ptt_check), true, None);
        let mode_continuous = MenuItem::new(format!("Continuous{}", cont_check), true, None);

        let mode_ptt_id = mode_ptt.id().clone();
        let mode_continuous_id = mode_continuous.id().clone();

        let mode_submenu = Submenu::new("Mode", true);
        mode_submenu.append(&mode_ptt)?;
        mode_submenu.append(&mode_continuous)?;

        let hotkey_item = MenuItem::new("Hotkey...", true, None);
        let hotkey_id = hotkey_item.id().clone();

        let models_item = MenuItem::new("Models...", true, None);
        let models_id = models_item.id().clone();

        let exit_item = MenuItem::new("Quit", true, None);
        let exit_id = exit_item.id().clone();

        let menu = Menu::new();
        menu.append(&mode_submenu)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&hotkey_item)?;
        menu.append(&models_item)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&exit_item)?;

        let tray = TrayIconBuilder::new()
            .with_tooltip("SpeechVox - Idle")
            .with_icon(icon)
            .with_menu(Box::new(menu))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create tray icon: {}", e))?;

        Ok(Self {
            _tray: tray,
            exit_id,
            mode_ptt_id,
            mode_continuous_id,
            hotkey_id,
            models_id,
        })
    }
}

fn load_png_icon(png_data: &[u8]) -> Result<Icon> {
    let img = image::load_from_memory(png_data)
        .map_err(|e| anyhow::anyhow!("Failed to decode PNG: {}", e))?;

    let img = img.resize_exact(32, 32, image::imageops::FilterType::Lanczos3);
    let (width, height) = img.dimensions();
    let rgba = img.to_rgba8().into_raw();

    Icon::from_rgba(rgba, width, height)
        .map_err(|e| anyhow::anyhow!("Failed to create icon: {}", e))
}
