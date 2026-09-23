#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

#define PIN_PHASE 2
#define ARENA 32768

alignas(16) static uint8_t tensor_arena[ARENA];

void halt() {
  while (true) {
    digitalWrite(PIN_PHASE, HIGH); delay(80);
    digitalWrite(PIN_PHASE, LOW);  delay(80);
  }
}

void setup() {
  pinMode(PIN_PHASE, OUTPUT);
  digitalWrite(PIN_PHASE, LOW);

  const tflite::Model* model = tflite::GetModel(kws_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) halt();

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, ARENA);
  if (interpreter.AllocateTensors() != kTfLiteOk) halt();

  TfLiteTensor* input = interpreter.input(0);
  for (int i = 0; i < (int)input->bytes; i++) input->data.int8[i] = 0;

  while (true) {
    digitalWrite(PIN_PHASE, LOW);
    delay(4000);

    digitalWrite(PIN_PHASE, HIGH);
    uint32_t t0 = millis();
    while (millis() - t0 < 4000) {
      interpreter.Invoke();
    }
  }
}

void loop() {}