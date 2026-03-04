#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod audio;
mod config;
mod engine;
mod ffi;
mod hotkey_settings;
mod hotkeys;
mod model_manager;
mod models;
mod overlay;
mod tray;

use crate::config::{Config, TranscriptionMode};
use crate::engine::{Engine, EngineCommand, EngineResult};
use crate::hotkeys::{check_hotkey_event, HotkeyAction, HotkeyManager};
use crate::overlay::{AppStatus, Overlay};
use crate::tray::TrayManager;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tao::event::{ElementState, Event, MouseButton, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder};
use tray_icon::menu::{ContextMenu, Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tracing::{error, info, warn};

#[derive(Debug)]
enum UserEvent {
    HotkeyPressed,
    HotkeyReleased,
    TrayQuit,
    TrayModeChanged(TranscriptionMode),
    TrayModels,
    OpenHotkeySettings,
    ReloadHotkey,
}

fn main() {
    // Check for --hotkeys FIRST, before mutex (hotkey settings is a separate mode)
    if std::env::args().any(|a| a == "--hotkeys") {
        hotkey_settings::run();
        return;
    }

    // Check for --models, before mutex (model manager is a separate mode)
    if std::env::args().any(|a| a == "--models") {
        model_manager::run();
        return;
    }

    // Log to file so we can see output even with windows_subsystem = "windows"
    let log_path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.join("speechvox.log")))
        .unwrap_or_else(|| std::path::PathBuf::from("speechvox.log"));
    let log_file = std::fs::File::create(&log_path).ok();

    if let Some(file) = log_file {
        tracing_subscriber::fmt()
            .with_env_filter("speechvox=debug")
            .with_writer(file)
            .with_ansi(false)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter("speechvox=debug")
            .init();
    }

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

    // First-launch: no model configured — open model manager and wait
    if config.effective_model_dir().is_none() {
        info!("No model configured, launching model manager");
        if let Ok(exe) = std::env::current_exe() {
            let _ = std::process::Command::new(&exe).arg("--models").status();
            // Reload config after model manager exits
            config = Config::load().unwrap_or_else(|e| {
                warn!("Failed to reload config: {}, using defaults", e);
                Config::default()
            });
            if config.effective_model_dir().is_none() {
                info!("No model selected, exiting");
                return;
            }
        }
    }

    // Set up hotkey via global-hotkey crate
    let mut hotkey_manager = match HotkeyManager::from_config(&config.hotkey) {
        Ok(hm) => hm,
        Err(e) => {
            error!("Failed to register hotkey '{}': {}", config.hotkey, e);
            // Fall back to F9
            warn!("Falling back to F9 hotkey");
            config.hotkey = "F9".to_string();
            HotkeyManager::from_config("F9").expect("F9 hotkey must register")
        }
    };
    // Store the hotkey ID in an atomic so the listener thread can see reloads
    let hotkey_id = Arc::new(std::sync::atomic::AtomicU32::new(hotkey_manager.hotkey_id()));

    let mut current_mode = config.mode;
    info!("Transcription mode: {:?}", current_mode);

    // Create event loop with custom user events
    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    let proxy = event_loop.create_proxy();

    // Create overlay
    let mut overlay = match Overlay::new(&event_loop, config.overlay_x, config.overlay_y) {
        Ok(o) => o,
        Err(e) => {
            error!("Failed to create overlay: {}", e);
            return;
        }
    };

    // Create tray
    let tray = match TrayManager::new(current_mode) {
        Ok(t) => t,
        Err(e) => {
            error!("Failed to create tray: {}", e);
            return;
        }
    };

    // Create overlay right-click context menu (mirrors tray menu)
    let ctx_ptt = MenuItem::new("Push-to-Talk", true, None);
    let ctx_continuous = MenuItem::new("Continuous", true, None);
    let ctx_hotkey = MenuItem::new("Hotkey...", true, None);
    let ctx_models = MenuItem::new("Models...", true, None);
    let ctx_quit = MenuItem::new("Quit", true, None);
    let ctx_ptt_id = ctx_ptt.id().clone();
    let ctx_continuous_id = ctx_continuous.id().clone();
    let ctx_hotkey_id = ctx_hotkey.id().clone();
    let ctx_models_id = ctx_models.id().clone();
    let ctx_quit_id = ctx_quit.id().clone();

    let ctx_menu = Menu::new();
    let _ = ctx_menu.append(&ctx_ptt);
    let _ = ctx_menu.append(&ctx_continuous);
    let _ = ctx_menu.append(&PredefinedMenuItem::separator());
    let _ = ctx_menu.append(&ctx_hotkey);
    let _ = ctx_menu.append(&ctx_models);
    let _ = ctx_menu.append(&PredefinedMenuItem::separator());
    let _ = ctx_menu.append(&ctx_quit);

    // Start engine
    let model_path = config
        .effective_model_dir()
        .expect("model dir resolved above")
        .to_string_lossy()
        .to_string();
    let (engine, engine_rx) = Engine::new(model_path);

    // Audio capture
    let mut audio = match audio::AudioCapture::new() {
        Ok(a) => a,
        Err(e) => {
            error!("Failed to init audio: {}", e);
            return;
        }
    };

    let running = Arc::new(AtomicBool::new(true));

    // Spawn hotkey listener thread (using global-hotkey crate)
    let proxy_hotkey = proxy.clone();
    let running_hotkey = Arc::clone(&running);
    let hotkey_id_clone = Arc::clone(&hotkey_id);
    std::thread::Builder::new()
        .name("hotkey-listener".to_string())
        .spawn(move || {
            let receiver = HotkeyManager::receiver();
            info!("Hotkey listener started");
            while running_hotkey.load(Ordering::SeqCst) {
                if let Ok(event) = receiver.recv_timeout(Duration::from_millis(100)) {
                    let current_id = hotkey_id_clone.load(Ordering::SeqCst);
                    if let Some(action) = check_hotkey_event(&event, current_id) {
                        let user_event = match action {
                            HotkeyAction::Pressed => UserEvent::HotkeyPressed,
                            HotkeyAction::Released => UserEvent::HotkeyReleased,
                        };
                        info!("Hotkey {:?}", action);
                        let _ = proxy_hotkey.send_event(user_event);
                    }
                }
            }
        })
        .expect("Failed to spawn hotkey listener thread");

    // Spawn tray menu listener thread
    let exit_id = tray.exit_id.clone();
    let mode_ptt_id = tray.mode_ptt_id.clone();
    let mode_continuous_id = tray.mode_continuous_id.clone();
    let tray_hotkey_id = tray.hotkey_id.clone();
    let tray_models_id = tray.models_id.clone();
    let ctx_ptt_id_clone = ctx_ptt_id.clone();
    let ctx_continuous_id_clone = ctx_continuous_id.clone();
    let ctx_hotkey_id_clone = ctx_hotkey_id.clone();
    let ctx_models_id_clone = ctx_models_id.clone();
    let ctx_quit_id_clone = ctx_quit_id.clone();
    let proxy_menu = proxy.clone();
    let running_menu = Arc::clone(&running);
    std::thread::Builder::new()
        .name("menu-listener".to_string())
        .spawn(move || {
            let menu_receiver = MenuEvent::receiver().clone();
            while running_menu.load(Ordering::SeqCst) {
                if let Ok(event) = menu_receiver.recv_timeout(Duration::from_millis(100)) {
                    let id = event.id();
                    if id == &exit_id || id == &ctx_quit_id_clone {
                        let _ = proxy_menu.send_event(UserEvent::TrayQuit);
                    } else if id == &mode_ptt_id || id == &ctx_ptt_id_clone {
                        let _ = proxy_menu.send_event(UserEvent::TrayModeChanged(TranscriptionMode::PushToTalk));
                    } else if id == &mode_continuous_id || id == &ctx_continuous_id_clone {
                        let _ = proxy_menu.send_event(UserEvent::TrayModeChanged(TranscriptionMode::ContinuousStreaming));
                    } else if id == &tray_hotkey_id || id == &ctx_hotkey_id_clone {
                        let _ = proxy_menu.send_event(UserEvent::OpenHotkeySettings);
                    } else if id == &tray_models_id || id == &ctx_models_id_clone {
                        let _ = proxy_menu.send_event(UserEvent::TrayModels);
                    }
                }
            }
        })
        .expect("Failed to spawn menu listener thread");

    let _tray = tray;

    let mut state = AppStatus::Loading;
    let overlay_window_id = overlay.window_id();

    // Handle for the continuous forwarding thread (Mode 3)
    let mut continuous_stop: Option<Arc<AtomicBool>> = None;

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
                running.store(false, Ordering::SeqCst);
                stop_continuous_mode(&mut continuous_stop, &mut audio, &engine);
                save_and_quit(&mut config, current_mode, &mut overlay, &engine, control_flow);
            }

            Event::WindowEvent {
                event:
                    WindowEvent::MouseInput {
                        state: ElementState::Pressed,
                        button: MouseButton::Left,
                        ..
                    },
                window_id,
                ..
            } if window_id == overlay_window_id => {
                overlay.start_drag();
            }

            Event::WindowEvent {
                event:
                    WindowEvent::MouseInput {
                        state: ElementState::Pressed,
                        button: MouseButton::Right,
                        ..
                    },
                window_id,
                ..
            } if window_id == overlay_window_id => {
                // Update menu text to show current mode with * marker
                let ptt_label = if current_mode == TranscriptionMode::PushToTalk {
                    "Push-to-Talk *"
                } else {
                    "Push-to-Talk"
                };
                let cont_label = if current_mode == TranscriptionMode::ContinuousStreaming {
                    "Continuous *"
                } else {
                    "Continuous"
                };
                ctx_ptt.set_text(ptt_label);
                ctx_continuous.set_text(cont_label);
                ctx_menu.show_context_menu_for_hwnd(overlay.hwnd(), None);
            }

            Event::UserEvent(user_event) => match user_event {
                UserEvent::HotkeyPressed => {
                    match current_mode {
                        TranscriptionMode::PushToTalk => {
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
                        TranscriptionMode::ContinuousStreaming => {
                            if state == AppStatus::Idle {
                                info!("Hotkey pressed - start continuous listening");
                                match audio.start_continuous() {
                                    Ok(chunk_rx) => {
                                        engine.send(EngineCommand::StartContinuous);

                                        // Spawn forwarding thread
                                        let stop_flag = Arc::new(AtomicBool::new(false));
                                        let stop_clone = Arc::clone(&stop_flag);
                                        let engine_tx = engine.cmd_tx();
                                        std::thread::Builder::new()
                                            .name("audio-forward".to_string())
                                            .spawn(move || {
                                                while !stop_clone.load(Ordering::SeqCst) {
                                                    match chunk_rx.recv_timeout(Duration::from_millis(100)) {
                                                        Ok(chunk) => {
                                                            if engine_tx.send(EngineCommand::FeedAudio(chunk)).is_err() {
                                                                break;
                                                            }
                                                        }
                                                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                                                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                                                    }
                                                }
                                            })
                                            .expect("Failed to spawn audio-forward thread");

                                        continuous_stop = Some(stop_flag);
                                        state = AppStatus::Listening;
                                        overlay.set_status(AppStatus::Listening);
                                    }
                                    Err(e) => {
                                        error!("Failed to start continuous capture: {}", e);
                                    }
                                }
                            } else if state == AppStatus::Listening {
                                // Stop continuous listening
                                info!("Hotkey pressed - stop continuous listening");
                                stop_continuous_mode(&mut continuous_stop, &mut audio, &engine);
                                state = AppStatus::Processing;
                                overlay.set_status(AppStatus::Processing);
                            }
                        }
                    }
                }

                UserEvent::HotkeyReleased => {
                    match current_mode {
                        TranscriptionMode::PushToTalk => {
                            if state == AppStatus::Recording {
                                info!("Hotkey released - stop recording");
                                let samples = audio.stop_recording();
                                if samples.len() > 1600 {
                                    state = AppStatus::Processing;
                                    overlay.set_status(AppStatus::Processing);
                                    engine.send(EngineCommand::Transcribe(samples));
                                } else {
                                    warn!("Recording too short, ignoring");
                                    state = AppStatus::Idle;
                                    overlay.set_status(AppStatus::Idle);
                                }
                            }
                        }
                        TranscriptionMode::ContinuousStreaming => {
                            // No-op on release in continuous mode
                        }
                    }
                }

                UserEvent::TrayModeChanged(new_mode) => {
                    if new_mode != current_mode {
                        info!("Mode changed: {:?} -> {:?}", current_mode, new_mode);
                        if state == AppStatus::Listening {
                            stop_continuous_mode(&mut continuous_stop, &mut audio, &engine);
                        }
                        current_mode = new_mode;
                        config.mode = new_mode;
                        let _ = config.save();
                        if state != AppStatus::Loading && state != AppStatus::Processing {
                            state = AppStatus::Idle;
                            overlay.set_status(AppStatus::Idle);
                        }
                    }
                }

                UserEvent::TrayModels => {
                    // Fire-and-forget: open model manager as child process
                    if let Ok(exe) = std::env::current_exe() {
                        let _ = std::process::Command::new(&exe)
                            .arg("--models")
                            .spawn();
                        info!("Launched model manager");
                    }
                }

                UserEvent::OpenHotkeySettings => {
                    // Spawn child process, wait for it to exit, then reload hotkey
                    if let Ok(exe) = std::env::current_exe() {
                        let proxy_reload = proxy.clone();
                        std::thread::Builder::new()
                            .name("hotkey-settings".to_string())
                            .spawn(move || {
                                let _ = std::process::Command::new(&exe)
                                    .arg("--hotkeys")
                                    .status(); // blocks until window closes
                                let _ = proxy_reload.send_event(UserEvent::ReloadHotkey);
                            })
                            .ok();
                        info!("Launched hotkey settings");
                    }
                }

                UserEvent::ReloadHotkey => {
                    // Reload hotkey from config after settings window closed
                    if let Ok(new_cfg) = Config::load() {
                        if new_cfg.hotkey != config.hotkey {
                            match hotkey_manager.reload(&new_cfg.hotkey) {
                                Ok(()) => {
                                    hotkey_id.store(hotkey_manager.hotkey_id(), Ordering::SeqCst);
                                    config.hotkey = new_cfg.hotkey;
                                    info!("Hotkey reloaded successfully");
                                }
                                Err(e) => {
                                    error!("Failed to reload hotkey: {}", e);
                                }
                            }
                        }
                    }
                }

                UserEvent::TrayQuit => {
                    running.store(false, Ordering::SeqCst);
                    stop_continuous_mode(&mut continuous_stop, &mut audio, &engine);
                    save_and_quit(&mut config, current_mode, &mut overlay, &engine, control_flow);
                }
            },

            Event::MainEventsCleared => {
                // Poll engine results
                while let Ok(result) = engine_rx.try_recv() {
                    match result {
                        EngineResult::ModelReady => {
                            info!("Model loaded, ready for transcription");
                            state = AppStatus::Idle;
                            overlay.set_status(AppStatus::Idle);
                        }
                        EngineResult::ModelError(e) => {
                            error!("Model loading failed: {}", e);
                        }
                        EngineResult::TranscriptionDone(text) => {
                            info!("Transcription: '{}'", text);
                            if !text.is_empty() {
                                type_text(&text);
                            }
                            state = AppStatus::Idle;
                            overlay.set_status(AppStatus::Idle);
                        }
                        EngineResult::TranscriptionChunk(text) => {
                            if !text.is_empty() {
                                type_text(&text);
                            }
                            // Stay in Listening state for continuous mode
                        }
                        EngineResult::TranscriptionError(e) => {
                            error!("Transcription error: {}", e);
                            state = AppStatus::Idle;
                            overlay.set_status(AppStatus::Idle);
                        }
                        EngineResult::ContinuousStarted => {
                            info!("Continuous streaming active");
                        }
                        EngineResult::ContinuousStopped => {
                            info!("Continuous streaming ended");
                            if state == AppStatus::Processing || state == AppStatus::Listening {
                                state = AppStatus::Idle;
                                overlay.set_status(AppStatus::Idle);
                            }
                        }
                    }
                }

                // Update RMS while recording or listening
                if state == AppStatus::Recording || state == AppStatus::Listening {
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

fn stop_continuous_mode(
    stop_flag: &mut Option<Arc<AtomicBool>>,
    audio: &mut audio::AudioCapture,
    engine: &Engine,
) {
    if let Some(flag) = stop_flag.take() {
        flag.store(true, Ordering::SeqCst);
    }
    audio.stop_continuous();
    engine.send(EngineCommand::StopContinuous);
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
    current_mode: TranscriptionMode,
    overlay: &mut Overlay,
    engine: &Engine,
    control_flow: &mut ControlFlow,
) {
    let (x, y) = overlay.get_position();
    config.overlay_x = Some(x);
    config.overlay_y = Some(y);
    config.mode = current_mode;
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
