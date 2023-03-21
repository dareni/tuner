use core::mem::MaybeUninit;
use cpal::traits::{DeviceTrait, HostTrait};
use cpal::Stream;
use log::debug;
use realfft::RealFftPlanner;
use realfft::RealToComplex;
use ringbuf::Consumer;
use ringbuf::HeapRb;
use ringbuf::SharedRb;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use std::sync::Arc;

struct FftControl {
    planner: Arc<dyn RealToComplex<f32>>,
    indata1: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
}

impl FftControl {
    fn new(total_samples: usize) -> FftControl {
        let mut realplanner = RealFftPlanner::<f32>::new();
        let r2c_planner = realplanner.plan_fft_forward(total_samples);
        let buf1 = r2c_planner.make_input_vec();
        let buf3 = r2c_planner.make_output_vec();

        FftControl {
            planner: r2c_planner,
            indata1: buf1,
            spectrum: buf3,
        }
    }
}

pub struct AudioInput {
    pub stream_config: cpal::StreamConfig,
    pub consumer_buf: Consumer<f32, Arc<SharedRb<f32, Vec<MaybeUninit<f32>>>>>,
    pub input_stream: Stream,
}

impl AudioInput {
    pub fn err_fn(err: cpal::StreamError) {
        eprintln!("an error occurred on stream: {}", err);
    }

    pub fn new() -> AudioInput {
        let host = cpal::default_host();
        let input_device = host.default_input_device().expect("Input device error");
        let stream_config: cpal::StreamConfig = input_device
            .default_input_config()
            .expect("Default config create error.")
            .into();

        // The buffer to share samples
        let ring = HeapRb::<f32>::new(
            ((stream_config.channels as u32) * stream_config.sample_rate.0 * 2) as usize,
        );

        let (mut producer, consumer_buf) = ring.split();
        let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut output_fell_behind = false;
            for &sample in data {
                if producer.push(sample).is_err() {
                    output_fell_behind = true;
                }
            }
            if output_fell_behind {
                eprintln!("output stream fell behind: try increasing latency");
            }
        };

        let input_stream: Stream = input_device
            .build_input_stream(&stream_config, input_data_fn, AudioInput::err_fn, None)
            .unwrap();

        AudioInput {
            stream_config,
            consumer_buf,
            input_stream,
        }
    }
}

fn main() {}

pub fn old() {
    let hz = 2637.0;

    let sample_interval_sec = 0.00016;
    let total_samples = 6000;

    let period = 1.0 / hz;
    let step: f32 = (PI * 2.0 * sample_interval_sec / period) as f32;
    let samples_per_cycle = period / sample_interval_sec;

    if samples_per_cycle < 2.0 {
        panic!("Sample rate too low!");
    }

    let mut fft_control = FftControl::new(total_samples);

    //Create sine wave data and store it.
    generate_sin(total_samples, &mut fft_control.indata1, step);
    fft_control
        .planner
        .process(&mut fft_control.indata1, &mut fft_control.spectrum)
        .unwrap();
    let bin_freq = get_bin_frequency(sample_interval_sec as f32, total_samples as f32);
    let _max_freq = get_dominate_freq(&mut fft_control.spectrum, bin_freq);
}

fn get_bin_frequency(sample_interval_sec: f32, total_samples: f32) -> f32 {
    //The 1st bin is D.C. ie 0hz.
    //The 2nd bin is:     sample_freq / total_samples
    //The nth bin is: n * sample_freq / total_samples
    let freq_bin = 1.0 / sample_interval_sec as f32 / total_samples as f32;
    freq_bin
}

fn get_dominate_freq(spectrum: &mut Vec<Complex<f32>>, bin_freq: f32) -> f32 {
    let max_peak = spectrum
        .iter()
        .enumerate()
        .max_by_key(|&(_, freq)| freq.norm() as u32);

    if let Some((i, _)) = max_peak {
        i as f32 * bin_freq
    } else {
        0_f32
    }
}

pub fn get_dominate_freq_old(spectrum: &mut Vec<Complex<f32>>, bin_freq: f32) -> f32 {
    let mut count = 0;
    let mut max_freq = 0_f32;
    let mut max_amplitude = 0_f32;
    for amplitude in spectrum.into_iter() {
        let freq = count as f32 * bin_freq;
        debug!("   val:{} freq: {}hz", amplitude.norm(), freq);
        let norm_amplitude = amplitude.norm();
        if norm_amplitude > max_amplitude {
            max_amplitude = norm_amplitude;
            max_freq = freq;
            debug!("   val:{} freq: {}hz", amplitude.norm(), max_freq);
        }
        count += 1;
    }
    max_freq
}

fn generate_sin(total_samples: usize, indata: &mut Vec<f32>, step: f32) {
    for count in 0..total_samples {
        indata[count] = (step * (count as f32)).sin();
        debug!("Sample {}: {}", count, indata[count]);
    }
}

#[cfg(test)]
pub mod tests {
    use crate::generate_sin;
    use crate::get_bin_frequency;
    use crate::get_dominate_freq;
    use crate::FftControl;

    use rustfft::{num_complex::Complex, FftPlanner};
    use std::f64::consts::PI;
    use std::time::SystemTime;

    #[test]
    pub fn test_rust_fft() {
        //Test the duration of the direct rustfft implementation.
        //Config of rustfft by realfft make the fft operation more efficient.
        println!("Test rustfft lib");
        let sample_interval_sec = 0.00016;
        let total_samples = 6000;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(total_samples);
        let hz = 196.0;
        let period = 1.0 / hz;
        let step: f32 = (PI * 2.0 * sample_interval_sec / period) as f32;

        let time0 = SystemTime::now();
        let mut signal = (0..total_samples)
            .map(|x| Complex::new((step * x as f32).sin() as f32, 0_f32))
            .collect::<Vec<_>>();
        let time1 = SystemTime::now();
        fft.process(&mut signal);
        let time2 = SystemTime::now();
        let max_peak = signal
            .iter()
            .take(total_samples / 2)
            .enumerate()
            .max_by_key(|&(_, freq)| freq.norm() as u32);

        let samples_per_cycle = period / sample_interval_sec;
        let bin_freq = get_bin_frequency(sample_interval_sec as f32, total_samples as f32);
        let dom_freq = match max_peak {
            Some((i, _)) => Some(i as f32 * bin_freq),
            None => None,
        };
        let time3 = SystemTime::now();
        print_stats(
            dom_freq.unwrap(),
            total_samples,
            sample_interval_sec,
            hz,
            samples_per_cycle,
            bin_freq,
            time0,
            time1,
            time2,
            time3,
        );

        assert_eq!((hz as f32), dom_freq.unwrap().round());
    }

    #[test]
    pub fn test_real_fft() {
        //Test each open string frequency and the highest note E7.
        //Adjust the asserts to allow for 1hz inaccuracy.

        //G3 = 196hz
        //D4 = 293.7hz
        //A4 = 440hz period of 2.27ms
        //E5 = 659.3hz
        //E7 = 2637hz

        let sample_interval_sec = 0.00016;
        let total_samples = 6000;
        let mut fft_control = FftControl::new(total_samples);
        let bin_freq = get_bin_frequency(sample_interval_sec as f32, total_samples as f32);

        let hz = 196.0;
        let period = 1.0 / hz;
        let step: f32 = (PI * 2.0 * sample_interval_sec / period) as f32;
        generate_sin(total_samples, &mut fft_control.indata1, step);
        fft_control
            .planner
            .process(&mut fft_control.indata1, &mut fft_control.spectrum)
            .unwrap();
        let max_freq = get_dominate_freq(&mut fft_control.spectrum, bin_freq);
        assert_eq!((hz as f32), max_freq.round());

        let hz = 293.7;
        let period = 1.0 / hz;
        let step: f32 = (PI * 2.0 * sample_interval_sec / period) as f32;
        generate_sin(total_samples, &mut fft_control.indata1, step);
        fft_control
            .planner
            .process(&mut fft_control.indata1, &mut fft_control.spectrum)
            .unwrap();
        let max_freq = get_dominate_freq(&mut fft_control.spectrum, bin_freq);
        assert_eq!((hz as f32).round(), max_freq.round());

        let hz = 440.0;
        let period = 1.0 / hz;
        let step: f32 = (PI * 2.0 * sample_interval_sec / period) as f32;
        generate_sin(total_samples, &mut fft_control.indata1, step);
        fft_control
            .planner
            .process(&mut fft_control.indata1, &mut fft_control.spectrum)
            .unwrap();
        let max_freq = get_dominate_freq(&mut fft_control.spectrum, bin_freq);
        assert_eq!((hz as f32).round(), max_freq.round());

        let hz = 659.3;
        let period = 1.0 / hz;
        let step: f32 = (PI * 2.0 * sample_interval_sec / period) as f32;
        generate_sin(total_samples, &mut fft_control.indata1, step);
        fft_control
            .planner
            .process(&mut fft_control.indata1, &mut fft_control.spectrum)
            .unwrap();
        let max_freq = get_dominate_freq(&mut fft_control.spectrum, bin_freq);
        assert_eq!((hz as f32).round(), max_freq.round());

        let hz = 2637.0;
        let period = 1.0 / hz;
        let step: f32 = (PI * 2.0 * sample_interval_sec / period) as f32;
        let time0 = SystemTime::now();
        generate_sin(total_samples, &mut fft_control.indata1, step);
        let time1 = SystemTime::now();
        fft_control
            .planner
            .process(&mut fft_control.indata1, &mut fft_control.spectrum)
            .unwrap();
        let time2 = SystemTime::now();
        let max_freq = get_dominate_freq(&mut fft_control.spectrum, bin_freq);
        let time3 = SystemTime::now();
        assert_eq!((hz as f32), max_freq.trunc());
        let samples_per_cycle = period / sample_interval_sec;
        print_stats(
            max_freq,
            total_samples,
            sample_interval_sec,
            hz,
            samples_per_cycle,
            bin_freq,
            time0,
            time1,
            time2,
            time3,
        );
    }

    fn print_stats(
        max_freq: f32,
        total_samples: usize,
        sample_interval_sec: f64,
        hz: f64,
        samples_per_cycle: f64,
        bin_freq: f32,
        time0: SystemTime,
        time1: SystemTime,
        time2: SystemTime,
        time3: SystemTime,
    ) {
        println!("==> Dominate Freq: {}hz", max_freq);
        println!("Total samples: {}", total_samples);
        println!("Sample interval: {}s", sample_interval_sec);
        println!(
            "Total sample : {}s",
            sample_interval_sec * (total_samples as f64)
        );
        println!("Test Freq: (hz) {}", hz);
        println!("Samples/cycle: {}", samples_per_cycle);
        println!("Bucket freq: {}", bin_freq);
        println!();
        println!("Durations");
        println!(
            "Data Population: {}s",
            time1.duration_since(time0).unwrap().as_secs_f32()
        );
        println!(
            "Fft Calculation: {}s",
            time2.duration_since(time1).unwrap().as_secs_f32()
        );
        println!(
            "Result Analysis: {}s",
            time3.duration_since(time2).unwrap().as_secs_f32()
        );
    }

    #[test]
    fn play_440() {
        use cpal::platform::Host;
        use cpal::traits::StreamTrait;
        use cpal::traits::{DeviceTrait, HostTrait};
        use cpal::Stream;
        use cpal::SupportedStreamConfig;
        let host: Host = cpal::default_host();
        let device: cpal::Device = Host::default_output_device(&host).expect("no output device");
        let default_config: SupportedStreamConfig =
            DeviceTrait::default_output_config(&device).expect("no default config");
        let stream_config = default_config.config();
        let sample_rate = stream_config.sample_rate.0;
        let channels = stream_config.channels as usize;

        // Produce a sinusoid of maximum amplitude.
        let mut sample_clock = 0;
        let mut next_value = move || {
            sample_clock = (sample_clock + 1) % sample_rate;
            (sample_clock as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin()
        };

        let stream: Stream = DeviceTrait::build_output_stream(
            &device,
            &stream_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                for frame in data.chunks_mut(channels) {
                    let value: f32 = cpal::Sample::from_sample(next_value());
                    //let value: f32 = cpal::Sample::from::<f32>(&next_value());
                    for sample in frame.iter_mut() {
                        *sample = value;
                    }
                }
            },
            move |err| {
                eprintln!("An error occurred on the output audio stream: {}", err);
            },
            None,
        )
        .expect("Failed to create stream!");

        StreamTrait::play(&stream).expect("Error on stream playback.");
        println!("press ctrl-c to end");

        std::thread::sleep(std::time::Duration::from_secs(2));
    }

    fn error_fn(err: cpal::StreamError) {
        eprintln!("An error occurred on stream: {}", err);
    }

    #[test]
    fn record_n_play() {
        use cpal::platform::Host;
        //use cpal::traits::StreamTrait;
        use cpal::traits::StreamTrait;
        use cpal::traits::{DeviceTrait, HostTrait};
        //use cpal::Stream;
        //use cpal::SupportedStreamConfig;
        use ringbuf::HeapRb;

        let host: Host = cpal::default_host();
        let input_device = host.default_input_device().expect("Input device error");
        let default_input_stream_config: cpal::SupportedStreamConfig = input_device
            .default_input_config()
            .expect("Stream config error");
        let stream_config: cpal::StreamConfig = default_input_stream_config.into();

        //let device: cpal::Device = Host::default_input_device(&host).expect("no input device");
        //let default_config: SupportedStreamConfig = DeviceTrait::default_input_config(&device).expect("no default config");
        //let stream_config = default_input_stream_config.config();
        //let recording_time_sec = 2;
        //let sample_rate = stream_config.sample_rate.0;
        //let channels = stream_config.channels as usize;

        // The buffer to share samples
        //let ring = HeapRb::<f32>::new(recording_time_sec as usize * sample_rate as usize * 2);
        //let (mut producer, mut consumer) = ring.split();

        //let recorder = move |data: &[f32], _: &_| {
        //for frame in data.chunks(channels) {
        //    for sample in frame.iter() {
        //        let value: f32 = cpal::Sample::to_sample(*sample);
        //        producer.push(value).expect("Error on pushing input data.");
        //    }
        //}
        //for &sample in data {
        //    producer.push(sample).expect("blah");
        //}
        //};

        let latency = 150_f32;
        let latency_frames = (latency / 1_000.0) * stream_config.sample_rate.0 as f32;
        let latency_samples = latency_frames as usize * stream_config.channels as usize;
        // The buffer to share samples
        let ring = HeapRb::<f32>::new(latency_samples * 4);
        let (mut producer, consumer) = ring.split();

        // Fill the samples with 0.0 equal to the length of the delay.
        //for _ in 0..latency_samples {
        for _ in 0..10 {
            //The ring buffer has twice as much space as necessary to add latency here,
            //so this should never fail
            producer.push(0.42_f32).unwrap();
        }

        let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut output_fell_behind = false;
            for &sample in data {
                if producer.push(sample).is_err() {
                    output_fell_behind = true;
                }
            }
            if output_fell_behind {
                eprintln!("output stream fell behind: try increasing latency");
            }
        };

        let input_stream = input_device
            .build_input_stream(&stream_config, input_data_fn, error_fn, None)
            .expect("Input stream creation error.");

        //recorder(&vec![1_f32,2_f32,3_f32]);

        //let channels = &channels;
        //println!("{}",channels);
        //recorder(vec![1.0,2.0,3.0].as_ref());
        //        let mut process =  move |data: &[f32]| {recorder(data)};

        //      let stream: Stream = DeviceTrait::build_input_stream(
        //          &device,
        //          &stream_config,
        //          move |data: &[f32], _: &_| recorder(data),
        //          move |err| {
        //              eprintln!("An error occurred on the output audio stream: {}", err);
        //          },
        //          Some(std::time::Duration::from_secs(recording_time_sec)),
        //      )
        //      .expect("Failed to create stream!");

        input_stream.play().expect("Play input stream error.");

        //StreamTrait::play(&stream).expect("Error on record.");
        std::thread::sleep(std::time::Duration::from_millis(1000));
        input_stream.pause().expect("Pause input stream error.");
        std::thread::sleep(std::time::Duration::from_millis(100));

        println!();
        println!();
        let mut count = 0;
        for x in consumer.iter() {
            println!("{},buf:{}", count, x);
            count = count + 1;
        }
        println!();
        println!();
    }

    /////////////////////////////////////////////////////

    #[test]
    fn input_test() {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        use ringbuf::HeapRb;

        let latency = 150_f32;
        let host = cpal::default_host();

        let input_device = host.default_input_device().expect("Input device error");
        let output_device = host.default_output_device().expect("out device error");
        // We'll try and use the same configuration between streams to keep it simple.
        let config: cpal::StreamConfig = input_device
            .default_input_config()
            .expect("Default config create error.")
            .into();

        // Create a delay in case the input and output devices aren't synced.
        let latency_frames = (latency / 1_000.0) * config.sample_rate.0 as f32;
        let latency_samples = latency_frames as usize * config.channels as usize;
        println!(
            "Sample Rate:{}, No Channels:{}",
            config.sample_rate.0, config.channels
        );

        // The buffer to share samples
        let ring = HeapRb::<f32>::new(latency_samples * 2);
        let (mut producer, mut consumer) = ring.split();

        // Fill the samples with 0.0 equal to the length of the delay.
        for _ in 0..latency_samples {
            // The ring buffer has twice as much space as necessary to add latency here,
            // so this should never fail
            producer.push(0.0).unwrap();
        }

        let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut output_fell_behind = false;
            for &sample in data {
                if producer.push(sample).is_err() {
                    output_fell_behind = true;
                }
            }
            if output_fell_behind {
                eprintln!("output stream fell behind: try increasing latency");
            }
        };

        let output_data_fn = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let mut input_fell_behind = false;
            for sample in data {
                *sample = match consumer.pop() {
                    Some(s) => s,
                    None => {
                        input_fell_behind = true;
                        0.0
                    }
                };
            }
            if input_fell_behind {
                eprintln!("input stream fell behind: try increasing latency");
            }
        };

        // Build streams.
        println!(
            "Attempting to build both streams with f32 samples and `{:?}`.",
            config
        );
        let input_stream = input_device
            .build_input_stream(&config, input_data_fn, err_fn, None)
            .unwrap();
        let output_stream = output_device
            .build_output_stream(&config, output_data_fn, err_fn, None)
            .unwrap();
        println!("Successfully built streams.");

        // Play the streams.
        println!(
            "Starting the input and output streams with `{}` milliseconds of latency.",
            latency
        );

        input_stream.play().unwrap();
        output_stream.play().unwrap();

        // Run for 3 seconds before closing.
        println!("Playing for 3 seconds... ");

        std::thread::sleep(std::time::Duration::from_millis(3000));

        drop(input_stream);
        drop(output_stream);
        println!("Done!");
    }

    fn err_fn(err: cpal::StreamError) {
        eprintln!("an error occurred on stream: {}", err);
    }

    #[test]
    fn audio_test() {
        use crate::AudioInput;
        use cpal::traits::StreamTrait;

        let audio_input = AudioInput::new();
        audio_input.input_stream.play().unwrap();

        std::thread::sleep(std::time::Duration::from_millis(500));

        //optimium sample interval 6250Hz
        //optimium samples = 6000
        let sample_rate = audio_input.stream_config.sample_rate.0 as f32;
        let sample_factor: f32 = sample_rate / 6250_f32.ceil();
        let sample_interval_sec = sample_factor / sample_rate;
        let total_samples = 6000;

        let mut fft_control = FftControl::new(total_samples);

        let mut count = 0;

        for x in audio_input
            .consumer_buf
            .iter()
            .step_by(audio_input.stream_config.channels as usize)
        {
            fft_control.indata1[count] = *x;
            count += 1;
            if count == total_samples {
                fft_control
                    .planner
                    .process(&mut fft_control.indata1, &mut fft_control.spectrum)
                    .unwrap();
                let bin_freq = get_bin_frequency(sample_interval_sec as f32, total_samples as f32);
                let max_freq = get_dominate_freq(&mut fft_control.spectrum, bin_freq);
                println!("{}Hz", max_freq);
                count = 0;
            }
        }
    }
}
