#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include "test_data.h"
#include <stdio.h>

constexpr int kArenaSize = 64 * 1024;
alignas(16) static uint8_t tensor_arena[kArenaSize];

void halt(const char* msg) { Serial.println(msg); while (true) delay(1000); }

char buf[160];
void P(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  Serial.print(buf);
}

void setup() {
  Serial.begin(115200);
  Serial.println("BOOT");
  delay(3000);
  delay(500);
  Serial.println("\n== KWS int8 model test on Pico ==");

  const tflite::Model* model = tflite::GetModel(kws_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) halt("Schema version mismatch");

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) halt("AllocateTensors failed");

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  P("input bytes: %d (expect %d), int8: %s\n", (int)input->bytes, INPUT_SIZE,
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
    for (int c = 1; c < N_CLASSES; c++)
      if (output->data.int8[c] > output->data.int8[best]) best = c;

    same_as_laptop += (best == expected_pred[i]);
    correct += (best == test_labels[i]);
    total_us += dt;
    if (dt > max_us) max_us = dt;
    P("#%02d true=%d laptop=%d pico=%d %lu us\n", i,
      test_labels[i], expected_pred[i], best, (unsigned long)dt);
  }

  Serial.println("---------------- RESULT ----------------");
  P("Same answer as laptop: %d / %d\n", same_as_laptop, N_TEST);
  P("Correct vs true label: %d / %d\n", correct, N_TEST);
  P("Latency avg: %.1f ms, max: %.1f ms\n", total_us / (float)N_TEST / 1000.0f, max_us / 1000.0f);
  P("Arena actually used: %u bytes\n", (unsigned)interpreter.arena_used_bytes());
}

void loop() {}