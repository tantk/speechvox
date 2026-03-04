use anyhow::Result;
use global_hotkey::{
    hotkey::{Code, HotKey, Modifiers},
    GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState,
};
use tracing::info;

pub struct HotkeyManager {
    manager: GlobalHotKeyManager,
    current_hotkey: HotKey,
    hotkey_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeyAction {
    Pressed,
    Released,
}

impl HotkeyManager {
    pub fn from_config(hotkey_str: &str) -> Result<Self> {
        let manager = GlobalHotKeyManager::new()
            .map_err(|e| anyhow::anyhow!("Failed to create hotkey manager: {}", e))?;

        let hotkey = parse_hotkey(hotkey_str)?;
        let hotkey_id = hotkey.id();

        manager
            .register(hotkey)
            .map_err(|e| anyhow::anyhow!("Failed to register hotkey: {}", e))?;

        info!("Hotkey registered: {}", format_hotkey_display(hotkey_str));

        Ok(Self {
            manager,
            current_hotkey: hotkey,
            hotkey_id,
        })
    }

    pub fn hotkey_id(&self) -> u32 {
        self.hotkey_id
    }

    /// Unregister old hotkey and register new one from config string.
    pub fn reload(&mut self, hotkey_str: &str) -> Result<()> {
        let _ = self.manager.unregister(self.current_hotkey);

        let hotkey = parse_hotkey(hotkey_str)?;
        self.manager
            .register(hotkey)
            .map_err(|e| anyhow::anyhow!("Failed to register hotkey: {}", e))?;

        self.current_hotkey = hotkey;
        self.hotkey_id = hotkey.id();
        info!("Hotkey reloaded: {}", format_hotkey_display(hotkey_str));
        Ok(())
    }

    pub fn receiver() -> crossbeam_channel::Receiver<GlobalHotKeyEvent> {
        GlobalHotKeyEvent::receiver().clone()
    }
}

pub fn check_hotkey_event(event: &GlobalHotKeyEvent, hotkey_id: u32) -> Option<HotkeyAction> {
    if event.id == hotkey_id {
        match event.state {
            HotKeyState::Pressed => Some(HotkeyAction::Pressed),
            HotKeyState::Released => Some(HotkeyAction::Released),
        }
    } else {
        None
    }
}

/// Parse a hotkey string like "Control+F9" or "F8" into a HotKey.
fn parse_hotkey(s: &str) -> Result<HotKey> {
    let parts: Vec<&str> = s.split('+').collect();

    let mut modifiers = Modifiers::empty();
    let mut key_code: Option<Code> = None;

    for part in parts {
        let part = part.trim();
        match part.to_lowercase().as_str() {
            "control" | "ctrl" => modifiers |= Modifiers::CONTROL,
            "alt" => modifiers |= Modifiers::ALT,
            "shift" => modifiers |= Modifiers::SHIFT,
            "super" | "win" | "meta" => modifiers |= Modifiers::SUPER,
            _ => {
                key_code = Some(parse_key_code(part)?);
            }
        }
    }

    let code = key_code.ok_or_else(|| anyhow::anyhow!("No key code found in hotkey string: {}", s))?;
    let mods = if modifiers.is_empty() { None } else { Some(modifiers) };
    Ok(HotKey::new(mods, code))
}

fn parse_key_code(s: &str) -> Result<Code> {
    let code = match s {
        "Backquote" | "`" => Code::Backquote,
        "Digit1" | "1" => Code::Digit1,
        "Digit2" | "2" => Code::Digit2,
        "Digit3" | "3" => Code::Digit3,
        "Digit4" | "4" => Code::Digit4,
        "Digit5" | "5" => Code::Digit5,
        "Digit6" | "6" => Code::Digit6,
        "Digit7" | "7" => Code::Digit7,
        "Digit8" | "8" => Code::Digit8,
        "Digit9" | "9" => Code::Digit9,
        "Digit0" | "0" => Code::Digit0,
        "KeyA" | "A" => Code::KeyA,
        "KeyB" | "B" => Code::KeyB,
        "KeyC" | "C" => Code::KeyC,
        "KeyD" | "D" => Code::KeyD,
        "KeyE" | "E" => Code::KeyE,
        "KeyF" | "F" => Code::KeyF,
        "KeyG" | "G" => Code::KeyG,
        "KeyH" | "H" => Code::KeyH,
        "KeyI" | "I" => Code::KeyI,
        "KeyJ" | "J" => Code::KeyJ,
        "KeyK" | "K" => Code::KeyK,
        "KeyL" | "L" => Code::KeyL,
        "KeyM" | "M" => Code::KeyM,
        "KeyN" | "N" => Code::KeyN,
        "KeyO" | "O" => Code::KeyO,
        "KeyP" | "P" => Code::KeyP,
        "KeyQ" | "Q" => Code::KeyQ,
        "KeyR" | "R" => Code::KeyR,
        "KeyS" | "S" => Code::KeyS,
        "KeyT" | "T" => Code::KeyT,
        "KeyU" | "U" => Code::KeyU,
        "KeyV" | "V" => Code::KeyV,
        "KeyW" | "W" => Code::KeyW,
        "KeyX" | "X" => Code::KeyX,
        "KeyY" | "Y" => Code::KeyY,
        "KeyZ" | "Z" => Code::KeyZ,
        "Minus" | "-" => Code::Minus,
        "Equal" | "Equals" | "=" => Code::Equal,
        "BracketLeft" | "OpenBracket" | "[" => Code::BracketLeft,
        "BracketRight" | "CloseBracket" | "]" => Code::BracketRight,
        "Backslash" | "\\" => Code::Backslash,
        "Semicolon" | ";" => Code::Semicolon,
        "Quote" | "'" => Code::Quote,
        "Comma" | "," => Code::Comma,
        "Period" | "." => Code::Period,
        "Slash" | "/" => Code::Slash,
        "F1" => Code::F1,
        "F2" => Code::F2,
        "F3" => Code::F3,
        "F4" => Code::F4,
        "F5" => Code::F5,
        "F6" => Code::F6,
        "F7" => Code::F7,
        "F8" => Code::F8,
        "F9" => Code::F9,
        "F10" => Code::F10,
        "F11" => Code::F11,
        "F12" => Code::F12,
        "Enter" => Code::Enter,
        "Backspace" => Code::Backspace,
        "Space" => Code::Space,
        "Tab" => Code::Tab,
        "CapsLock" => Code::CapsLock,
        "Escape" | "Esc" => Code::Escape,
        "Insert" => Code::Insert,
        "Delete" | "Del" => Code::Delete,
        "Home" => Code::Home,
        "End" => Code::End,
        "PageUp" => Code::PageUp,
        "PageDown" => Code::PageDown,
        "ArrowUp" | "Up" => Code::ArrowUp,
        "ArrowDown" | "Down" => Code::ArrowDown,
        "ArrowLeft" | "Left" => Code::ArrowLeft,
        "ArrowRight" | "Right" => Code::ArrowRight,
        "Numpad0" => Code::Numpad0,
        "Numpad1" => Code::Numpad1,
        "Numpad2" => Code::Numpad2,
        "Numpad3" => Code::Numpad3,
        "Numpad4" => Code::Numpad4,
        "Numpad5" => Code::Numpad5,
        "Numpad6" => Code::Numpad6,
        "Numpad7" => Code::Numpad7,
        "Numpad8" => Code::Numpad8,
        "Numpad9" => Code::Numpad9,
        "NumpadAdd" => Code::NumpadAdd,
        "NumpadSubtract" => Code::NumpadSubtract,
        "NumpadMultiply" => Code::NumpadMultiply,
        "NumpadDivide" => Code::NumpadDivide,
        "NumpadEnter" => Code::NumpadEnter,
        "NumpadDecimal" => Code::NumpadDecimal,
        _ => return Err(anyhow::anyhow!("Unknown key code: {}", s)),
    };
    Ok(code)
}

/// Format hotkey for display (more user-friendly).
pub fn format_hotkey_display(s: &str) -> String {
    s.replace("Control", "Ctrl")
        .replace("Backquote", "`")
        .replace("Minus", "-")
        .replace("Equal", "=")
        .replace("BracketLeft", "[")
        .replace("BracketRight", "]")
        .replace("Backslash", "\\")
        .replace("Semicolon", ";")
        .replace("Quote", "'")
        .replace("Comma", ",")
        .replace("Period", ".")
        .replace("Slash", "/")
        .replace("Key", "")
        .replace("Digit", "")
}
