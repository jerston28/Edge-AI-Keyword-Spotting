#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include "test_data.h"

constexpr int kArenaSize = 24 * 1024;
alignas(16) static uint8_t tensor_arena[kArenaSize];

const char* names[8] = {"down", "go", "left", "no", "right", "stop", "up", "yes"};

void halt(const char* msg) {
  Serial.println(msg);
  while (true) delay(1000);
}

void setup() {
  Serial.begin(115200);
  delay(3000);
  Serial.println("\n== KWS int8 model test on ESP32 ==");

  const tflite::Model* model = tflite::GetModel(kws_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) halt("Schema version mismatch");

  static tflite::MicroMutableOpResolver<4> resolver;
resolver.AddConv2D();
resolver.AddDepthwiseConv2D();
resolver.AddMean();
resolver.AddFullyConnected();
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk)
    halt("AllocateTensors failed (arena too small or unsupported op)");

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  Serial.printf("input bytes: %d (expect %d), int8: %s\n", (int)input->bytes, INPUT_SIZE,
                input->type == kTfLiteInt8 ? "yes" : "NO");
  if ((int)input->bytes != INPUT_SIZE) halt("Input size mismatch");

  int same_as_laptop = 0, correct = 0;
  uint32_t total_us = 0, max_us = 0;

  for (int i = 0; i < N_TEST; i++) {
    memcpy(input->data.int8, test_inputs[i], INPUT_SIZE);
    uint32_t t0 = micros();
    TfLiteStatus s = interpreter.Invoke();
    uint32_t dt = micros() - t0;
    if (s != kTfLiteOk) halt("Invoke failed");

    int best = 0;
    for (int c = 1; c < 8; c++)
      if (output->data.int8[c] > output->data.int8[best]) best = c;

    same_as_laptop += (best == expected_pred[i]);
    correct += (best == test_labels[i]);
    total_us += dt;
    if (dt > max_us) max_us = dt;
    Serial.printf("#%02d true=%-5s laptop=%-5s esp32=%-5s %lu us\n", i,
                  names[test_labels[i]], names[expected_pred[i]], names[best], (unsigned long)dt);
  }

  Serial.println("---------------- RESULT ----------------");
  Serial.printf("Same answer as laptop: %d / %d\n", same_as_laptop, N_TEST);
  Serial.printf("Correct vs true label: %d / %d\n", correct, N_TEST);
  Serial.printf("Latency avg: %.1f ms, max: %.1f ms\n", total_us / (float)N_TEST / 1000.0f, max_us / 1000.0f);
  Serial.printf("Arena actually used: %u bytes\n", (unsigned)interpreter.arena_used_bytes());
}

void loop() {}