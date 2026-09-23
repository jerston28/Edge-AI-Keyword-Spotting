#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("== Operations used by this model ==");

  const tflite::Model* model = tflite::GetModel(kws_model);
  auto* opcodes = model->operator_codes();

  for (unsigned int i = 0; i < opcodes->size(); i++) {
    auto* code = opcodes->Get(i);
    int32_t builtin = code->deprecated_builtin_code();
    if (builtin == 127) builtin = code->builtin_code();
    Serial.print(i);
    Serial.print(": ");
    Serial.println(tflite::EnumNameBuiltinOperator((tflite::BuiltinOperator)builtin));
  }
  Serial.println("== done ==");
}

void loop() {}