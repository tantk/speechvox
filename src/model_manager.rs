use crate::config::Config;
use crate::models::{self, ModelEntry, ModelManifest};
use eframe::egui;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct DownloadProgress {
    file_name: String,
    bytes_downloaded: u64,
    bytes_total: u64,
    done: bool,
    error: Option<String>,
    cancelled: bool,
}

struct DownloadHandle {
    progress: Arc<Mutex<DownloadProgress>>,
    cancel: Arc<Mutex<bool>>,
}

struct ModelManagerApp {
    manifest: ModelManifest,
    config: Config,
    active_download: Option<(String, DownloadHandle)>,
    status_message: Option<String>,
}

impl ModelManagerApp {
    fn new() -> Self {
        let manifest = models::load_manifest();
        let config = Config::load().unwrap_or_default();
        Self {
            manifest,
            config,
            active_download: None,
            status_message: None,
        }
    }

    fn start_download(&mut self, model: &ModelEntry) {
        if self.active_download.is_some() {
            return;
        }

        let model_id = model.id.clone();
        let model_dir = models::model_dir(&model_id);
        let _ = std::fs::create_dir_all(&model_dir);

        let progress = Arc::new(Mutex::new(DownloadProgress {
            file_name: String::new(),
            bytes_downloaded: 0,
            bytes_total: model.total_download_size(),
            done: false,
            error: None,
            cancelled: false,
        }));
        let cancel = Arc::new(Mutex::new(false));

        let handle = DownloadHandle {
            progress: Arc::clone(&progress),
            cancel: Arc::clone(&cancel),
        };

        let files = model.files.clone();
        let hf_repo = self.manifest.hf_repo.clone();

        std::thread::Builder::new()
            .name("model-download".to_string())
            .spawn(move || {
                download_model_files(&hf_repo, &files, &model_dir, &progress, &cancel);
            })
            .expect("Failed to spawn download thread");

        self.active_download = Some((model_id, handle));
    }

    fn cancel_download(&mut self) {
        if let Some((_, ref handle)) = self.active_download {
            if let Ok(mut c) = handle.cancel.lock() {
                *c = true;
            }
        }
    }

    fn select_model(&mut self, model_id: &str) {
        self.config.active_model = Some(model_id.to_string());
        if let Err(e) = self.config.save() {
            self.status_message = Some(format!("Failed to save config: {}", e));
        } else {
            self.status_message = Some(format!("Active model set to: {}", model_id));
        }
    }
}

impl eframe::App for ModelManagerApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        // Poll download progress
        let mut download_finished_id: Option<String> = None;
        if let Some((ref model_id, ref handle)) = self.active_download {
            if let Ok(prog) = handle.progress.lock() {
                if prog.done || prog.error.is_some() || prog.cancelled {
                    download_finished_id = Some(model_id.clone());
                }
            }
            ctx.request_repaint();
        }
        if let Some(id) = download_finished_id {
            if let Some((_, handle)) = self.active_download.take() {
                if let Ok(prog) = handle.progress.lock() {
                    if prog.done {
                        self.status_message = Some(format!("{} downloaded successfully!", id));
                    } else if prog.cancelled {
                        self.status_message = Some("Download cancelled".to_string());
                    } else if let Some(ref e) = prog.error {
                        self.status_message = Some(format!("Download error: {}", e));
                    }
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui: &mut egui::Ui| {
            ui.heading("SpeechVox \u{2014} Model Manager");
            ui.add_space(8.0);

            let active_model = self.config.active_model.clone();
            let downloading_id = self.active_download.as_ref().map(|(id, _)| id.clone());

            for model in &self.manifest.models.clone() {
                let is_complete = models::is_model_complete(&model.id);
                let is_active = active_model.as_deref() == Some(&model.id);
                let is_downloading = downloading_id.as_deref() == Some(&model.id);

                egui::Frame::group(ui.style()).show(ui, |ui: &mut egui::Ui| {
                    ui.horizontal(|ui: &mut egui::Ui| {
                        if model.recommended {
                            ui.label(egui::RichText::new("\u{2605}").color(egui::Color32::GOLD));
                        }
                        ui.label(
                            egui::RichText::new(&model.name)
                                .strong()
                                .size(16.0),
                        );
                        if model.recommended {
                            ui.label(
                                egui::RichText::new("Recommended")
                                    .small()
                                    .color(egui::Color32::from_rgb(100, 180, 100)),
                            );
                        }
                    });
                    ui.label(&model.description);
                    ui.label(format!(
                        "{} \u{00B7} ~{} GB VRAM",
                        format_bytes(model.total_download_size()),
                        model.vram_mb / 1000
                    ));

                    ui.horizontal(|ui: &mut egui::Ui| {
                        if is_complete {
                            ui.label(
                                egui::RichText::new("Ready \u{2713}")
                                    .color(egui::Color32::from_rgb(100, 200, 100)),
                            );
                            if is_active {
                                ui.label(
                                    egui::RichText::new("(Active)")
                                        .color(egui::Color32::from_rgb(100, 150, 255)),
                                );
                            } else if ui.button("Select").clicked() {
                                let id = model.id.clone();
                                self.select_model(&id);
                            }
                        } else if is_downloading {
                            ui.label("Downloading...");
                        } else {
                            let can_download = self.active_download.is_none();
                            if ui
                                .add_enabled(can_download, egui::Button::new("Download"))
                                .clicked()
                            {
                                let m = model.clone();
                                self.start_download(&m);
                            }
                        }
                    });
                });
                ui.add_space(4.0);
            }

            // Download progress bar
            if let Some((_, ref handle)) = self.active_download {
                if let Ok(prog) = handle.progress.lock() {
                    ui.separator();
                    ui.label(format!("Downloading: {}", prog.file_name));
                    let fraction = if prog.bytes_total > 0 {
                        prog.bytes_downloaded as f32 / prog.bytes_total as f32
                    } else {
                        0.0
                    };
                    ui.add(
                        egui::ProgressBar::new(fraction)
                            .text(format!(
                                "{:.0}% \u{2014} {} / {}",
                                fraction * 100.0,
                                format_bytes(prog.bytes_downloaded),
                                format_bytes(prog.bytes_total)
                            ))
                            .animate(true),
                    );
                }
                if ui.button("Cancel").clicked() {
                    self.cancel_download();
                }
            }

            // Status message
            if let Some(ref msg) = self.status_message {
                ui.add_space(8.0);
                ui.label(msg);
            }

            // Footer
            ui.add_space(16.0);
            ui.separator();
            ui.horizontal(|ui: &mut egui::Ui| {
                if let Some(ref id) = active_model {
                    ui.label(format!("Active model: {}", id));
                } else {
                    ui.label("No model selected");
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui: &mut egui::Ui| {
                    if ui.button("Close").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });

            // Hint: manual file placement
            let models_path = models::models_dir();
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(format!(
                    "Already have a .vqf file? Place it as consolidated.vqf in:\n{}/<model-id>/",
                    models_path.display()
                ))
                .small()
                .weak(),
            );
        });
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else {
        format!("{} bytes", bytes)
    }
}

fn download_model_files(
    hf_repo: &str,
    files: &[models::ModelFile],
    model_dir: &PathBuf,
    progress: &Arc<Mutex<DownloadProgress>>,
    cancel: &Arc<Mutex<bool>>,
) {
    let mut total_downloaded: u64 = 0;

    for file in files {
        // Check cancel
        if *cancel.lock().unwrap() {
            if let Ok(mut p) = progress.lock() {
                p.cancelled = true;
            }
            return;
        }

        let dest = model_dir.join(&file.name);
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            hf_repo, file.hf_path
        );

        if let Ok(mut p) = progress.lock() {
            p.file_name = file.name.clone();
        }

        // Check for partial download (resume support)
        let existing_size = if dest.exists() {
            std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };

        // Skip if already fully downloaded
        if existing_size == file.size_bytes {
            total_downloaded += file.size_bytes;
            if let Ok(mut p) = progress.lock() {
                p.bytes_downloaded = total_downloaded;
            }
            continue;
        }

        match download_file_with_resume(&url, &dest, existing_size, file.size_bytes, progress, cancel, total_downloaded) {
            Ok(bytes) => {
                total_downloaded += bytes;
            }
            Err(e) => {
                if let Ok(mut p) = progress.lock() {
                    p.error = Some(e);
                }
                return;
            }
        }
    }

    // Write .complete marker
    let marker = model_dir.join(".complete");
    if let Err(e) = std::fs::write(&marker, "") {
        if let Ok(mut p) = progress.lock() {
            p.error = Some(format!("Failed to write .complete marker: {}", e));
        }
        return;
    }

    if let Ok(mut p) = progress.lock() {
        p.done = true;
    }
}

fn download_file_with_resume(
    url: &str,
    dest: &PathBuf,
    existing_size: u64,
    total_file_size: u64,
    progress: &Arc<Mutex<DownloadProgress>>,
    cancel: &Arc<Mutex<bool>>,
    base_downloaded: u64,
) -> Result<u64, String> {
    use std::io::{Read, Write};

    let agent = ureq::Agent::new_with_defaults();

    let request = agent.get(url);
    let request = if existing_size > 0 {
        request.header("Range", &format!("bytes={}-", existing_size))
    } else {
        request
    };

    let response = request.call().map_err(|e| format!("HTTP request failed: {}", e))?;

    let status = response.status().as_u16();
    if status != 200 && status != 206 {
        return Err(format!("HTTP {}", status));
    }

    let mut file = if existing_size > 0 && status == 206 {
        std::fs::OpenOptions::new()
            .append(true)
            .open(dest)
            .map_err(|e| format!("Failed to open file for append: {}", e))?
    } else {
        std::fs::File::create(dest)
            .map_err(|e| format!("Failed to create file: {}", e))?
    };

    let start_offset = if status == 206 { existing_size } else { 0 };
    let mut downloaded = start_offset;
    let mut reader = response.into_body().into_reader();
    let mut buf = vec![0u8; 64 * 1024];

    loop {
        if *cancel.lock().unwrap() {
            if let Ok(mut p) = progress.lock() {
                p.cancelled = true;
            }
            return Err("Cancelled".to_string());
        }

        let n = reader.read(&mut buf).map_err(|e| format!("Read error: {}", e))?;
        if n == 0 {
            break;
        }

        file.write_all(&buf[..n])
            .map_err(|e| format!("Write error: {}", e))?;

        downloaded += n as u64;
        if let Ok(mut p) = progress.lock() {
            p.bytes_downloaded = base_downloaded + downloaded;
        }
    }

    Ok(total_file_size)
}

pub fn run() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("SpeechVox \u{2014} Model Manager")
            .with_inner_size([520.0, 600.0])
            .with_min_inner_size([400.0, 400.0]),
        ..Default::default()
    };

    let _ = eframe::run_native(
        "SpeechVox Model Manager",
        options,
        Box::new(|_cc| Ok(Box::new(ModelManagerApp::new()))),
    );
}
