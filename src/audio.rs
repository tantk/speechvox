use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use tracing::{debug, error, warn};

const TARGET_SAMPLE_RATE: u32 = 16000;

pub struct AudioCapture {
    device: Device,
    config: StreamConfig,
    recording: Arc<AtomicBool>,
    buffer: Arc<Mutex<Vec<f32>>>,
    rms: Arc<AtomicU32>,
    stream: Option<Stream>,
}

impl AudioCapture {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        debug!("Audio host: {:?}", host.id());

        let device = host
            .default_input_device()
            .context("No input device available")?;

        debug!("Input device: {:?}", device.name().unwrap_or_default());

        let supported_config = device
            .default_input_config()
            .context("Failed to get default input config")?;

        debug!("Default config: {:?}", supported_config);

        // Try to use 16kHz mono, fall back to device default
        let config = match device.supported_input_configs() {
            Ok(mut configs) => {
                let supports_16k = configs.any(|c| {
                    c.channels() >= 1
                        && c.min_sample_rate().0 <= TARGET_SAMPLE_RATE
                        && c.max_sample_rate().0 >= TARGET_SAMPLE_RATE
                });

                if supports_16k {
                    debug!("Using 16kHz mono");
                    StreamConfig {
                        channels: 1,
                        sample_rate: cpal::SampleRate(TARGET_SAMPLE_RATE),
                        buffer_size: cpal::BufferSize::Default,
                    }
                } else {
                    debug!(
                        "Device doesn't support 16kHz, using default: {}Hz {}ch",
                        supported_config.sample_rate().0,
                        supported_config.channels()
                    );
                    StreamConfig {
                        channels: supported_config.channels(),
                        sample_rate: supported_config.sample_rate(),
                        buffer_size: cpal::BufferSize::Default,
                    }
                }
            }
            Err(_) => {
                debug!("Using default config");
                StreamConfig {
                    channels: supported_config.channels(),
                    sample_rate: supported_config.sample_rate(),
                    buffer_size: cpal::BufferSize::Default,
                }
            }
        };

        Ok(Self {
            device,
            config,
            recording: Arc::new(AtomicBool::new(false)),
            buffer: Arc::new(Mutex::new(Vec::new())),
            rms: Arc::new(AtomicU32::new(0f32.to_bits())),
            stream: None,
        })
    }

    pub fn current_rms(&self) -> f32 {
        f32::from_bits(self.rms.load(Ordering::Relaxed))
    }

    pub fn start_recording(&mut self) -> Result<()> {
        if self.recording.load(Ordering::SeqCst) {
            return Ok(());
        }

        self.buffer.lock().clear();
        self.rms.store(0f32.to_bits(), Ordering::Relaxed);
        self.recording.store(true, Ordering::SeqCst);

        let buffer = Arc::clone(&self.buffer);
        let recording = Arc::clone(&self.recording);
        let rms = Arc::clone(&self.rms);
        let source_sample_rate = self.config.sample_rate.0;
        let channels = self.config.channels as usize;

        debug!(
            "Starting audio stream: {}Hz, {} channels",
            source_sample_rate, channels
        );

        let err_fn = |err| error!("Audio stream error: {}", err);

        let stream = match self.device.default_input_config()?.sample_format() {
            SampleFormat::F32 => self.device.build_input_stream(
                &self.config,
                move |data: &[f32], _| {
                    if recording.load(Ordering::SeqCst) {
                        let mono_data = convert_to_mono(data, channels);
                        let resampled =
                            resample(&mono_data, source_sample_rate, TARGET_SAMPLE_RATE);
                        // Update live RMS
                        if !resampled.is_empty() {
                            let rms_val = (resampled.iter().map(|x| x * x).sum::<f32>()
                                / resampled.len() as f32)
                                .sqrt();
                            rms.store(rms_val.to_bits(), Ordering::Relaxed);
                        }
                        buffer.lock().extend(resampled);
                    }
                },
                err_fn,
                None,
            )?,
            SampleFormat::I16 => self.device.build_input_stream(
                &self.config,
                move |data: &[i16], _| {
                    if recording.load(Ordering::SeqCst) {
                        let float_data: Vec<f32> =
                            data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                        let mono_data = convert_to_mono(&float_data, channels);
                        let resampled =
                            resample(&mono_data, source_sample_rate, TARGET_SAMPLE_RATE);
                        if !resampled.is_empty() {
                            let rms_val = (resampled.iter().map(|x| x * x).sum::<f32>()
                                / resampled.len() as f32)
                                .sqrt();
                            rms.store(rms_val.to_bits(), Ordering::Relaxed);
                        }
                        buffer.lock().extend(resampled);
                    }
                },
                err_fn,
                None,
            )?,
            SampleFormat::U16 => self.device.build_input_stream(
                &self.config,
                move |data: &[u16], _| {
                    if recording.load(Ordering::SeqCst) {
                        let float_data: Vec<f32> = data
                            .iter()
                            .map(|&s| (s as f32 / u16::MAX as f32) * 2.0 - 1.0)
                            .collect();
                        let mono_data = convert_to_mono(&float_data, channels);
                        let resampled =
                            resample(&mono_data, source_sample_rate, TARGET_SAMPLE_RATE);
                        if !resampled.is_empty() {
                            let rms_val = (resampled.iter().map(|x| x * x).sum::<f32>()
                                / resampled.len() as f32)
                                .sqrt();
                            rms.store(rms_val.to_bits(), Ordering::Relaxed);
                        }
                        buffer.lock().extend(resampled);
                    }
                },
                err_fn,
                None,
            )?,
            _ => return Err(anyhow::anyhow!("Unsupported sample format")),
        };

        stream.play()?;
        self.stream = Some(stream);

        Ok(())
    }

    pub fn stop_recording(&mut self) -> Vec<f32> {
        self.recording.store(false, Ordering::SeqCst);
        self.stream = None;

        let audio = std::mem::take(&mut *self.buffer.lock());

        if !audio.is_empty() {
            let max_val = audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
            debug!(
                "Audio captured: {} samples ({:.1}s), max={:.3}, rms={:.3}",
                audio.len(),
                audio.len() as f32 / TARGET_SAMPLE_RATE as f32,
                max_val,
                rms
            );

            if max_val < 0.01 {
                warn!("Audio level very low - check microphone!");
            }
        } else {
            warn!("No audio captured!");
        }

        audio
    }

    /// Start continuous capture. Returns a receiver of ~100ms audio chunks (1600 samples at 16kHz).
    pub fn start_continuous(&mut self) -> Result<crossbeam_channel::Receiver<Vec<f32>>> {
        if self.recording.load(Ordering::SeqCst) {
            return Err(anyhow::anyhow!("Already recording"));
        }

        self.recording.store(true, Ordering::SeqCst);
        self.rms.store(0f32.to_bits(), Ordering::Relaxed);

        let (tx, rx) = crossbeam_channel::bounded(32);
        let recording = Arc::clone(&self.recording);
        let rms = Arc::clone(&self.rms);
        let source_sample_rate = self.config.sample_rate.0;
        let channels = self.config.channels as usize;

        debug!(
            "Starting continuous audio stream: {}Hz, {} channels",
            source_sample_rate, channels
        );

        let err_fn = |err| error!("Audio stream error: {}", err);

        // Chunk size: ~100ms at 16kHz = 1600 samples
        const CHUNK_SIZE: usize = 1600;

        let stream = match self.device.default_input_config()?.sample_format() {
            SampleFormat::F32 => {
                let acc = Arc::new(Mutex::new(Vec::with_capacity(CHUNK_SIZE * 2)));
                let acc2 = Arc::clone(&acc);
                let tx2 = tx.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[f32], _| {
                        if !recording.load(Ordering::SeqCst) { return; }
                        let mono_data = convert_to_mono(data, channels);
                        let resampled = resample(&mono_data, source_sample_rate, TARGET_SAMPLE_RATE);
                        if !resampled.is_empty() {
                            let rms_val = (resampled.iter().map(|x| x * x).sum::<f32>()
                                / resampled.len() as f32).sqrt();
                            rms.store(rms_val.to_bits(), Ordering::Relaxed);
                        }
                        let mut buf = acc2.lock();
                        buf.extend(resampled);
                        while buf.len() >= CHUNK_SIZE {
                            let chunk: Vec<f32> = buf.drain(..CHUNK_SIZE).collect();
                            if tx2.send(chunk).is_err() { return; }
                        }
                    },
                    err_fn,
                    None,
                )?
            }
            SampleFormat::I16 => {
                let acc = Arc::new(Mutex::new(Vec::with_capacity(CHUNK_SIZE * 2)));
                let acc2 = Arc::clone(&acc);
                let tx2 = tx.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[i16], _| {
                        if !recording.load(Ordering::SeqCst) { return; }
                        let float_data: Vec<f32> = data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                        let mono_data = convert_to_mono(&float_data, channels);
                        let resampled = resample(&mono_data, source_sample_rate, TARGET_SAMPLE_RATE);
                        if !resampled.is_empty() {
                            let rms_val = (resampled.iter().map(|x| x * x).sum::<f32>()
                                / resampled.len() as f32).sqrt();
                            rms.store(rms_val.to_bits(), Ordering::Relaxed);
                        }
                        let mut buf = acc2.lock();
                        buf.extend(resampled);
                        while buf.len() >= CHUNK_SIZE {
                            let chunk: Vec<f32> = buf.drain(..CHUNK_SIZE).collect();
                            if tx2.send(chunk).is_err() { return; }
                        }
                    },
                    err_fn,
                    None,
                )?
            }
            SampleFormat::U16 => {
                let acc = Arc::new(Mutex::new(Vec::with_capacity(CHUNK_SIZE * 2)));
                let acc2 = Arc::clone(&acc);
                let tx2 = tx.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[u16], _| {
                        if !recording.load(Ordering::SeqCst) { return; }
                        let float_data: Vec<f32> = data.iter().map(|&s| (s as f32 / u16::MAX as f32) * 2.0 - 1.0).collect();
                        let mono_data = convert_to_mono(&float_data, channels);
                        let resampled = resample(&mono_data, source_sample_rate, TARGET_SAMPLE_RATE);
                        if !resampled.is_empty() {
                            let rms_val = (resampled.iter().map(|x| x * x).sum::<f32>()
                                / resampled.len() as f32).sqrt();
                            rms.store(rms_val.to_bits(), Ordering::Relaxed);
                        }
                        let mut buf = acc2.lock();
                        buf.extend(resampled);
                        while buf.len() >= CHUNK_SIZE {
                            let chunk: Vec<f32> = buf.drain(..CHUNK_SIZE).collect();
                            if tx2.send(chunk).is_err() { return; }
                        }
                    },
                    err_fn,
                    None,
                )?
            }
            _ => return Err(anyhow::anyhow!("Unsupported sample format")),
        };

        stream.play()?;
        self.stream = Some(stream);

        Ok(rx)
    }

    /// Stop continuous capture.
    pub fn stop_continuous(&mut self) {
        self.recording.store(false, Ordering::SeqCst);
        self.stream = None;
    }
}

fn convert_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return data.to_vec();
    }

    data.chunks(channels)
        .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
        .collect()
}

fn resample(data: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return data.to_vec();
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let new_len = (data.len() as f64 * ratio) as usize;
    let mut result = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f64 / ratio;
        let idx = src_idx as usize;
        let frac = src_idx - idx as f64;

        let sample = if idx + 1 < data.len() {
            data[idx] * (1.0 - frac as f32) + data[idx + 1] * frac as f32
        } else if idx < data.len() {
            data[idx]
        } else {
            0.0
        };

        result.push(sample);
    }

    result
}
