use anyhow::Result;
use image::GenericImageView;
use softbuffer::Surface;
use std::num::NonZeroU32;
use std::rc::Rc;
use tao::{
    dpi::{LogicalSize, PhysicalPosition},
    event_loop::EventLoopWindowTarget,
    platform::windows::WindowExtWindows,
    window::{Icon, Window, WindowBuilder},
};

const OVERLAY_WIDTH: u32 = 120;
const OVERLAY_HEIGHT: u32 = 50;
const WINDOWS_MINIMIZED_COORD_THRESHOLD: i32 = -30_000;
const WINDOW_ICON_PNG: &[u8] = include_bytes!("../assets/icon.png");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppStatus {
    Loading,
    Idle,
    Recording,
    Processing,
    Listening,
}

fn is_valid_saved_position(x: i32, y: i32) -> bool {
    x > WINDOWS_MINIMIZED_COORD_THRESHOLD && y > WINDOWS_MINIMIZED_COORD_THRESHOLD
}

fn load_window_icon() -> Option<Icon> {
    let img = image::load_from_memory(WINDOW_ICON_PNG).ok()?;
    let img = img.resize_exact(32, 32, image::imageops::FilterType::Lanczos3);
    let (width, height) = img.dimensions();
    let rgba = img.to_rgba8().into_raw();
    Icon::from_rgba(rgba, width, height).ok()
}

pub struct Overlay {
    window: Rc<Window>,
    surface: Surface<Rc<Window>, Rc<Window>>,
    #[allow(dead_code)]
    visible: bool,
    status: AppStatus,
    rms_level: f32,
    width: u32,
    height: u32,
}

impl Overlay {
    pub fn new<T>(
        event_loop: &EventLoopWindowTarget<T>,
        saved_x: Option<i32>,
        saved_y: Option<i32>,
    ) -> Result<Self> {
        let window = WindowBuilder::new()
            .with_title("Loading...")
            .with_inner_size(LogicalSize::new(OVERLAY_WIDTH as f64, OVERLAY_HEIGHT as f64))
            .with_decorations(false)
            .with_always_on_top(true)
            .with_window_icon(load_window_icon())
            .with_resizable(false)
            .build(event_loop)
            .map_err(|e| anyhow::anyhow!("Failed to create overlay window: {}", e))?;

        match (saved_x, saved_y) {
            (Some(x), Some(y)) if is_valid_saved_position(x, y) => {
                window.set_outer_position(PhysicalPosition::new(x, y));
            }
            _ => {
                if let Some(monitor) = window.primary_monitor() {
                    let monitor_size = monitor.size();
                    let scale = monitor.scale_factor();
                    let x = 20i32;
                    let y = ((monitor_size.height as f64 / scale) as i32 - 120).max(100);
                    window.set_outer_position(PhysicalPosition::new(x, y));
                }
            }
        }

        let window = Rc::new(window);
        let context = softbuffer::Context::new(window.clone())
            .map_err(|e| anyhow::anyhow!("Failed to create softbuffer context: {}", e))?;
        let surface = Surface::new(&context, window.clone())
            .map_err(|e| anyhow::anyhow!("Failed to create softbuffer surface: {}", e))?;

        let size = window.inner_size();

        let mut overlay = Self {
            window,
            surface,
            visible: true,
            status: AppStatus::Loading,
            rms_level: 0.0,
            width: size.width,
            height: size.height,
        };

        overlay.render();
        Ok(overlay)
    }

    #[allow(dead_code)]
    pub fn start_drag(&self) {
        let _ = self.window.drag_window();
    }

    pub fn get_position(&self) -> (i32, i32) {
        let pos = self
            .window
            .outer_position()
            .unwrap_or(PhysicalPosition::new(0, 0));
        (pos.x, pos.y)
    }

    pub fn set_status(&mut self, status: AppStatus) {
        self.status = status;

        let title = match status {
            AppStatus::Loading => "Loading...",
            AppStatus::Idle => "Idle",
            AppStatus::Recording => "Recording",
            AppStatus::Processing => "Processing...",
            AppStatus::Listening => "Listening...",
        };
        self.window.set_title(title);

        if status != AppStatus::Recording && status != AppStatus::Listening {
            self.rms_level = 0.0;
        }

        self.render();
    }

    pub fn set_rms(&mut self, rms: f32) {
        self.rms_level = rms.clamp(0.0, 1.0);
        if self.status == AppStatus::Recording || self.status == AppStatus::Listening {
            self.render();
        }
    }

    pub fn hwnd(&self) -> isize {
        self.window.hwnd()
    }

    pub fn window_id(&self) -> tao::window::WindowId {
        self.window.id()
    }

    pub fn handle_redraw(&mut self) {
        self.render();
    }

    fn render(&mut self) {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }

        self.width = size.width;
        self.height = size.height;

        if let (Some(w), Some(h)) = (NonZeroU32::new(self.width), NonZeroU32::new(self.height)) {
            let _ = self.surface.resize(w, h);
        } else {
            return;
        }

        let color = match self.status {
            AppStatus::Loading => 0xFF_DDAA00,    // Yellow
            AppStatus::Idle => 0xFF_505050,        // Dark gray
            AppStatus::Recording => 0xFF_DD3333,   // Red
            AppStatus::Processing => 0xFF_DDAA00,  // Yellow
            AppStatus::Listening => 0xFF_227744,   // Green
        };

        let border_color = match self.status {
            AppStatus::Loading => 0xFF_FFCC00,
            AppStatus::Idle => 0xFF_707070,
            AppStatus::Recording => 0xFF_FF5555,
            AppStatus::Processing => 0xFF_FFCC00,
            AppStatus::Listening => 0xFF_33BB66,
        };

        if let Ok(mut buffer) = self.surface.buffer_mut() {
            let w = self.width as usize;
            let h = self.height as usize;

            // Fill background
            for pixel in buffer.iter_mut() {
                *pixel = color;
            }

            // Draw border (1px)
            for x in 0..w {
                if x < buffer.len() {
                    buffer[x] = border_color;
                }
                if (h - 1) * w + x < buffer.len() {
                    buffer[(h - 1) * w + x] = border_color;
                }
            }
            for y in 0..h {
                if y * w < buffer.len() {
                    buffer[y * w] = border_color;
                }
                if y * w + w - 1 < buffer.len() {
                    buffer[y * w + w - 1] = border_color;
                }
            }

            // Draw RMS bar when recording or listening
            if (self.status == AppStatus::Recording || self.status == AppStatus::Listening)
                && self.rms_level > 0.001
            {
                let margin = 4usize;
                let bar_y_start = (h * 40 / 100).max(margin);
                let bar_y_end = (h * 60 / 100).min(h.saturating_sub(margin));
                let max_bar_width = w.saturating_sub(margin * 2);
                // Scale RMS: typical speech RMS is 0.01-0.3, map to 0-1
                let normalized = (self.rms_level * 5.0).clamp(0.0, 1.0);
                let bar_width = (normalized * max_bar_width as f32) as usize;

                let bar_color = 0xFF_FFFFFF; // White
                for y in bar_y_start..bar_y_end {
                    for x in margin..(margin + bar_width) {
                        let idx = y * w + x;
                        if idx < buffer.len() {
                            buffer[idx] = bar_color;
                        }
                    }
                }
            }

            let _ = buffer.present();
        }
    }
}
