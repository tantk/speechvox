use anyhow::Result;
use image::GenericImageView;
use tray_icon::{
    menu::{Menu, MenuId, MenuItem, PredefinedMenuItem},
    Icon, TrayIcon, TrayIconBuilder,
};

const ICON_GRAY: &[u8] = include_bytes!("../assets/mic_gray.png");

pub struct TrayManager {
    _tray: TrayIcon,
    pub gpu_toggle_id: MenuId,
    pub exit_id: MenuId,
}

impl TrayManager {
    pub fn new(gpu_enabled: bool) -> Result<Self> {
        let icon = load_png_icon(ICON_GRAY)?;

        let gpu_label = if gpu_enabled {
            "GPU: ON"
        } else {
            "GPU: OFF"
        };
        let gpu_toggle_item = MenuItem::new(gpu_label, true, None);
        let exit_item = MenuItem::new("Quit", true, None);

        let gpu_toggle_id = gpu_toggle_item.id().clone();
        let exit_id = exit_item.id().clone();

        let menu = Menu::new();
        menu.append(&gpu_toggle_item)?;
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
            gpu_toggle_id,
            exit_id,
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
