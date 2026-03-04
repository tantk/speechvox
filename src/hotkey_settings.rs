use eframe::egui;

#[derive(PartialEq)]
enum State {
    Idle,
    WaitingForKey,
}

struct HotkeySettingsApp {
    state: State,
    current_hotkey: String,
    captured_hotkey: Option<String>,
}

impl HotkeySettingsApp {
    fn new() -> Self {
        let config = crate::config::Config::load().unwrap_or_default();
        Self {
            state: State::Idle,
            current_hotkey: config.hotkey.clone(),
            captured_hotkey: None,
        }
    }
}

impl eframe::App for HotkeySettingsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Capture key events when waiting
        if self.state == State::WaitingForKey {
            ctx.input(|i| {
                for event in &i.events {
                    if let egui::Event::Key {
                        key,
                        pressed: true,
                        modifiers,
                        ..
                    } = event
                    {
                        // Ignore bare modifier keys
                        if matches!(
                            key,
                            egui::Key::ArrowUp | egui::Key::ArrowDown | egui::Key::ArrowLeft | egui::Key::ArrowRight
                            | egui::Key::F1 | egui::Key::F2 | egui::Key::F3 | egui::Key::F4
                            | egui::Key::F5 | egui::Key::F6 | egui::Key::F7 | egui::Key::F8
                            | egui::Key::F9 | egui::Key::F10 | egui::Key::F11 | egui::Key::F12
                            | egui::Key::A | egui::Key::B | egui::Key::C | egui::Key::D
                            | egui::Key::E | egui::Key::F | egui::Key::G | egui::Key::H
                            | egui::Key::I | egui::Key::J | egui::Key::K | egui::Key::L
                            | egui::Key::M | egui::Key::N | egui::Key::O | egui::Key::P
                            | egui::Key::Q | egui::Key::R | egui::Key::S | egui::Key::T
                            | egui::Key::U | egui::Key::V | egui::Key::W | egui::Key::X
                            | egui::Key::Y | egui::Key::Z
                            | egui::Key::Num0 | egui::Key::Num1 | egui::Key::Num2 | egui::Key::Num3
                            | egui::Key::Num4 | egui::Key::Num5 | egui::Key::Num6 | egui::Key::Num7
                            | egui::Key::Num8 | egui::Key::Num9
                            | egui::Key::Space | egui::Key::Escape | egui::Key::Tab
                            | egui::Key::Backspace | egui::Key::Enter | egui::Key::Insert
                            | egui::Key::Delete | egui::Key::Home | egui::Key::End
                            | egui::Key::PageUp | egui::Key::PageDown
                            | egui::Key::Backtick
                            | egui::Key::Minus | egui::Key::Equals
                            | egui::Key::OpenBracket | egui::Key::CloseBracket
                            | egui::Key::Backslash | egui::Key::Semicolon
                            | egui::Key::Quote | egui::Key::Comma | egui::Key::Period | egui::Key::Slash
                        ) {
                            let combo = format_combo(modifiers, key);
                            self.captured_hotkey = Some(combo);
                            self.state = State::Idle;
                        }
                    }
                }
            });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(20.0);
                ui.heading("Hotkey Settings");
                ui.add_space(20.0);

                ui.label(format!("Current hotkey: {}", display_hotkey(&self.current_hotkey)));
                ui.add_space(10.0);

                if let Some(ref captured) = self.captured_hotkey {
                    ui.label(format!("New hotkey: {}", display_hotkey(captured)));
                    ui.add_space(10.0);
                }

                if self.state == State::WaitingForKey {
                    ui.label("Press any key or key combo...");
                    ui.add_space(10.0);
                    if ui.button("Cancel capture").clicked() {
                        self.state = State::Idle;
                    }
                } else {
                    if ui.button("Change hotkey").clicked() {
                        self.state = State::WaitingForKey;
                        self.captured_hotkey = None;
                    }
                }

                ui.add_space(20.0);
                ui.separator();
                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    let can_save = self.captured_hotkey.is_some()
                        && self.captured_hotkey.as_deref() != Some(&self.current_hotkey);

                    if ui.add_enabled(can_save, egui::Button::new("Save")).clicked() {
                        if let Some(ref combo) = self.captured_hotkey {
                            if let Ok(mut config) = crate::config::Config::load() {
                                config.hotkey = combo.clone();
                                let _ = config.save();
                            }
                            self.current_hotkey = combo.clone();
                            self.captured_hotkey = None;
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    }

                    if ui.button("Cancel").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });
        });
    }
}

/// Build a combo string like "Control+Shift+F9" from egui modifiers + key.
fn format_combo(modifiers: &egui::Modifiers, key: &egui::Key) -> String {
    let mut parts = Vec::new();
    if modifiers.ctrl {
        parts.push("Control".to_string());
    }
    if modifiers.alt {
        parts.push("Alt".to_string());
    }
    if modifiers.shift {
        parts.push("Shift".to_string());
    }
    parts.push(key_to_config_name(key));
    parts.join("+")
}

/// Convert egui::Key to config-format name (e.g. "F9", "A", "Space").
fn key_to_config_name(key: &egui::Key) -> String {
    match key {
        egui::Key::F1 => "F1".into(),
        egui::Key::F2 => "F2".into(),
        egui::Key::F3 => "F3".into(),
        egui::Key::F4 => "F4".into(),
        egui::Key::F5 => "F5".into(),
        egui::Key::F6 => "F6".into(),
        egui::Key::F7 => "F7".into(),
        egui::Key::F8 => "F8".into(),
        egui::Key::F9 => "F9".into(),
        egui::Key::F10 => "F10".into(),
        egui::Key::F11 => "F11".into(),
        egui::Key::F12 => "F12".into(),
        egui::Key::A => "A".into(),
        egui::Key::B => "B".into(),
        egui::Key::C => "C".into(),
        egui::Key::D => "D".into(),
        egui::Key::E => "E".into(),
        egui::Key::F => "F".into(),
        egui::Key::G => "G".into(),
        egui::Key::H => "H".into(),
        egui::Key::I => "I".into(),
        egui::Key::J => "J".into(),
        egui::Key::K => "K".into(),
        egui::Key::L => "L".into(),
        egui::Key::M => "M".into(),
        egui::Key::N => "N".into(),
        egui::Key::O => "O".into(),
        egui::Key::P => "P".into(),
        egui::Key::Q => "Q".into(),
        egui::Key::R => "R".into(),
        egui::Key::S => "S".into(),
        egui::Key::T => "T".into(),
        egui::Key::U => "U".into(),
        egui::Key::V => "V".into(),
        egui::Key::W => "W".into(),
        egui::Key::X => "X".into(),
        egui::Key::Y => "Y".into(),
        egui::Key::Z => "Z".into(),
        egui::Key::Num0 => "0".into(),
        egui::Key::Num1 => "1".into(),
        egui::Key::Num2 => "2".into(),
        egui::Key::Num3 => "3".into(),
        egui::Key::Num4 => "4".into(),
        egui::Key::Num5 => "5".into(),
        egui::Key::Num6 => "6".into(),
        egui::Key::Num7 => "7".into(),
        egui::Key::Num8 => "8".into(),
        egui::Key::Num9 => "9".into(),
        egui::Key::Space => "Space".into(),
        egui::Key::Escape => "Escape".into(),
        egui::Key::Tab => "Tab".into(),
        egui::Key::Backspace => "Backspace".into(),
        egui::Key::Enter => "Enter".into(),
        egui::Key::Insert => "Insert".into(),
        egui::Key::Delete => "Delete".into(),
        egui::Key::Home => "Home".into(),
        egui::Key::End => "End".into(),
        egui::Key::PageUp => "PageUp".into(),
        egui::Key::PageDown => "PageDown".into(),
        egui::Key::ArrowUp => "Up".into(),
        egui::Key::ArrowDown => "Down".into(),
        egui::Key::ArrowLeft => "Left".into(),
        egui::Key::ArrowRight => "Right".into(),
        egui::Key::Backtick => "Backtick".into(),
        egui::Key::Minus => "Minus".into(),
        egui::Key::Equals => "Equals".into(),
        egui::Key::OpenBracket => "OpenBracket".into(),
        egui::Key::CloseBracket => "CloseBracket".into(),
        egui::Key::Backslash => "Backslash".into(),
        egui::Key::Semicolon => "Semicolon".into(),
        egui::Key::Quote => "Quote".into(),
        egui::Key::Comma => "Comma".into(),
        egui::Key::Period => "Period".into(),
        egui::Key::Slash => "Slash".into(),
        _ => format!("{:?}", key),
    }
}

/// Display-friendly version: "Control" → "Ctrl", etc.
fn display_hotkey(hotkey: &str) -> String {
    hotkey
        .replace("Control", "Ctrl")
}

pub fn run() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("SpeechVox \u{2014} Hotkey Settings")
            .with_inner_size([350.0, 250.0])
            .with_min_inner_size([300.0, 200.0]),
        ..Default::default()
    };

    let _ = eframe::run_native(
        "SpeechVox Hotkey Settings",
        options,
        Box::new(|_cc| Ok(Box::new(HotkeySettingsApp::new()))),
    );
}
