#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod audio;
mod config;
mod engine;
mod ffi;
mod overlay;
mod tray;

use crate::config::Config;
use crate::engine::{Engine, EngineResult};
use crate::overlay::{AppStatus, Overlay};
use crate::tray::TrayManager;

use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use std::time::{Duration, Instant};
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoop};
use tray_icon::menu::MenuEvent;
use tracing::{error, info, warn};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("speechvox=debug")
        .init();

    // Single-instance mutex
    let _mutex = match create_single_instance_mutex() {
        Some(m) => m,
        None => {
            error!("Another instance of SpeechVox is already running");
            return;
        }
    };

    info!("SpeechVox starting");

    // Load config
    let mut config = Config::load().unwrap_or_else(|e| {
        warn!("Failed to load config: {}, using defaults", e);
        Config::default()
    });
    let _ = config.validate();

    // Create event loop
    let event_loop = EventLoop::new();

    // Create overlay
    let mut overlay = match Overlay::new(&event_loop, config.overlay_x, config.overlay_y) {
        Ok(o) => o,
        Err(e) => {
            error!("Failed to create overlay: {}", e);
            return;
        }
    };

    // Create tray
    let tray = match TrayManager::new(config.use_gpu) {
        Ok(t) => t,
        Err(e) => {
            error!("Failed to create tray: {}", e);
            return;
        }
    };

    // Register hotkey (F9 by default)
    let hotkey_manager = GlobalHotKeyManager::new().expect("Failed to init hotkey manager");
    let hotkey = parse_hotkey(&config.hotkey);
    hotkey_manager.register(hotkey).expect("Failed to register hotkey");

    // Start engine
    let (engine, engine_rx) = Engine::new(
        config.model_path.clone(),
        config.mmproj_path.clone(),
        config.use_gpu,
    );

    // Audio capture
    let mut audio = match audio::AudioCapture::new() {
        Ok(a) => a,
        Err(e) => {
            error!("Failed to init audio: {}", e);
            return;
        }
    };

    let mut state = AppStatus::Loading;
    let overlay_window_id = overlay.window_id();
    let gpu_toggle_id = tray.gpu_toggle_id.clone();
    let exit_id = tray.exit_id.clone();

    info!("Entering event loop");

    event_loop.run(move |event, _target, control_flow| {
        // Poll at ~60fps for responsiveness
        *control_flow = ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(16));

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id,
                ..
            } if window_id == overlay_window_id => {
                save_and_quit(&mut config, &mut overlay, &engine, control_flow);
            }

            Event::WindowEvent {
                event: WindowEvent::CursorMoved { .. },
                window_id,
                ..
            } if window_id == overlay_window_id => {
                // Allow drag
            }

            Event::MainEventsCleared => {
                // Poll hotkey events
                if let Ok(event) = GlobalHotKeyEvent::receiver().try_recv() {
                    if event.id() == hotkey.id() {
                        match event.state() {
                            HotKeyState::Pressed => {
                                if state == AppStatus::Idle {
                                    info!("Hotkey pressed - start recording");
                                    if let Err(e) = audio.start_recording() {
                                        error!("Failed to start recording: {}", e);
                                    } else {
                                        state = AppStatus::Recording;
                                        overlay.set_status(AppStatus::Recording);
                                    }
                                }
                            }
                            HotKeyState::Released => {
                                if state == AppStatus::Recording {
                                    info!("Hotkey released - stop recording");
                                    let samples = audio.stop_recording();
                                    if samples.len() > 1600 {
                                        // At least 0.1s of audio
                                        state = AppStatus::Processing;
                                        overlay.set_status(AppStatus::Processing);
                                        engine.transcribe(samples);
                                    } else {
                                        warn!("Recording too short, ignoring");
                                        state = AppStatus::Idle;
                                        overlay.set_status(AppStatus::Idle);
                                    }
                                }
                            }
                        }
                    }
                }

                // Poll tray menu events
                if let Ok(event) = MenuEvent::receiver().try_recv() {
                    if event.id() == &exit_id {
                        save_and_quit(&mut config, &mut overlay, &engine, control_flow);
                        return;
                    } else if event.id() == &gpu_toggle_id {
                        config.use_gpu = !config.use_gpu;
                        info!("GPU toggled: {} (restart required)", config.use_gpu);
                        let _ = config.save();
                    }
                }

                // Poll engine results
                if let Ok(result) = engine_rx.try_recv() {
                    match result {
                        EngineResult::ModelReady => {
                            info!("Model loaded, ready for transcription");
                            state = AppStatus::Idle;
                            overlay.set_status(AppStatus::Idle);
                        }
                        EngineResult::ModelError(e) => {
                            error!("Model loading failed: {}", e);
                            // Stay in loading state but log the error
                        }
                        EngineResult::TranscriptionDone(text) => {
                            info!("Transcription: '{}'", text);
                            if !text.is_empty() {
                                type_text(&text);
                            }
                            state = AppStatus::Idle;
                            overlay.set_status(AppStatus::Idle);
                        }
                        EngineResult::TranscriptionError(e) => {
                            error!("Transcription error: {}", e);
                            state = AppStatus::Idle;
                            overlay.set_status(AppStatus::Idle);
                        }
                    }
                }

                // Update RMS while recording
                if state == AppStatus::Recording {
                    let rms = audio.current_rms();
                    overlay.set_rms(rms);
                }
            }

            Event::RedrawRequested(window_id) if window_id == overlay_window_id => {
                overlay.handle_redraw();
            }

            _ => {}
        }
    });
}

fn parse_hotkey(key_str: &str) -> global_hotkey::hotkey::HotKey {
    use global_hotkey::hotkey::{Code, HotKey};

    let code = match key_str.to_uppercase().as_str() {
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
        _ => {
            warn!("Unknown hotkey '{}', defaulting to F9", key_str);
            Code::F9
        }
    };

    HotKey::new(None, code)
}

fn type_text(text: &str) {
    // Small delay to let the user's key release register
    std::thread::sleep(Duration::from_millis(50));

    match enigo::Enigo::new(&enigo::Settings::default()) {
        Ok(mut enigo) => {
            use enigo::Keyboard;
            if let Err(e) = enigo.text(text) {
                error!("Failed to type text: {}", e);
            }
        }
        Err(e) => {
            error!("Failed to create enigo instance: {}", e);
        }
    }
}

fn save_and_quit(
    config: &mut Config,
    overlay: &mut Overlay,
    engine: &Engine,
    control_flow: &mut ControlFlow,
) {
    let (x, y) = overlay.get_position();
    config.overlay_x = Some(x);
    config.overlay_y = Some(y);
    let _ = config.save();
    engine.shutdown();
    *control_flow = ControlFlow::Exit;
}

fn create_single_instance_mutex() -> Option<SingleInstanceGuard> {
    use windows::core::s;
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::Threading::CreateMutexA;

    unsafe {
        let handle = CreateMutexA(None, true, s!("Global\\speechvox")).ok()?;
        let last_error = windows::Win32::Foundation::GetLastError();
        if last_error == windows::Win32::Foundation::ERROR_ALREADY_EXISTS {
            let _ = CloseHandle(handle);
            return None;
        }
        Some(SingleInstanceGuard(handle))
    }
}

struct SingleInstanceGuard(windows::Win32::Foundation::HANDLE);

impl Drop for SingleInstanceGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(self.0);
        }
    }
}
