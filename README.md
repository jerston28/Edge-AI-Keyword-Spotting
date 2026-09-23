# Edge AI Keyword Spotting Benchmark

A tiny 24K-parameter neural network that recognizes 8 spoken commands
(yes, no, up, down, left, right, stop, go) entirely on-device — no
cloud, no internet — benchmarked across an ESP32 and a Raspberry Pi
Pico.

## What it does

- Runs fully offline on a microcontroller
- Recognizes commands live through a microphone
- Reacts with an LED and a live web dashboard
- Benchmarks itself: model size, RAM, flash, latency, accuracy, and
  energy per inference

## How it works

1. **Feature extraction (MFCC):** raw audio is converted into a small
   49×10 "fingerprint" of the sound, computed identically on both the
   training laptop and the microcontroller (verified to match to
   0.00001 float precision).
2. **Model:** a depthwise-separable CNN (24,072 parameters) classifies
   the fingerprint into one of 10 classes (8 words + unknown +
   silence).
3. **Quantization:** the trained model is shrunk from 93.7 KB (float32)
   to 44.4 KB (int8) with a 0.6-point accuracy drop.
4. **On-device inference:** TensorFlow Lite Micro runs the int8 model
   on the microcontroller in real time, averaging predictions across
   two windows before triggering a detection.
5. **Live dashboard:** the ESP32 hosts a small web page over Wi-Fi
   showing the detected word, confidence, and latency in real time.

## Results

### Model
| Metric | Float32 | Int8 |
|---|---|---|
| Parameters | 24,072 | 24,072 |
| Size | 93.7 KB | 44.4 KB |
| Test accuracy | 92.8% | 92.2% |

### Fine-tuning on our own voice
Training only on public data (Google Speech Commands) confuses
similar-sounding words on a new mic/voice — e.g. "left" was
misheard as "yes." Adding an unknown/silence class and fine-tuning
on ~40 of our own recordings per word fixed this:

| | Old model (8-class) | New model (10-class) |
|---|---|---|
| Accuracy on our own voice | 91.9% | 98.6% |

### On-device benchmark
| | ESP32 | Raspberry Pi Pico |
|---|---|---|
| Latency | 183.0 ms → 181.1 ms (optimized) | 423.5 ms |
| RAM used | 22,572 bytes | 22,740 bytes |
| Flash used | 458,636 bytes (34%) | — |
| Energy / inference | ~64 mJ | not measured |

**Finding:** the ESP32's dual-core Xtensa runs the same int8 model
2.3x faster than the RP2040 Pico, despite using nearly identical RAM
— on this workload, speed is dominated by clock/architecture, not
memory.

### Optimization
Replacing TensorFlow's `AllOpsResolver` (every possible operation)
with a `MicroMutableOpResolver` containing only the 4 operations this
model actually uses (Conv2D, DepthwiseConv2D, Mean, FullyConnected)
cut flash usage by ~30% (655 KB → 459 KB) with zero accuracy loss and
a small latency improvement.

## Hardware
- ESP32 Dev Module
- Raspberry Pi Pico (RP2040)
- INMP441 I2S microphone
- INA219 current sensor (for energy measurement)
- Total cost: ~₹1400

## What I'd improve next
- Reduce false triggers on background noise with a larger/more varied
  "unknown" training set
- Try ESP-NN optimized kernels for faster inference
- Measure Pico's own energy consumption
- Expand beyond 8 fixed commands with a larger vocabulary

## Demo
[link to video]
