#include <Wire.h>
#include <Adafruit_INA219.h>

#define PIN_PHASE 6

Adafruit_INA219 ina219;

int last_phase = -1;
double sum_mW = 0;
long n = 0;
float last_idle = 0, last_busy = 0;

void setup() {
  Serial.begin(115200);
  delay(3000);
  Serial.println("== Power measurement ==");

  pinMode(PIN_PHASE, INPUT);
  Wire.begin();

  if (!ina219.begin()) {
    Serial.println("INA219 NOT FOUND. Check wiring.");
    while (true) delay(1000);
  }
  Serial.println("INA219 found. Logging per-phase averages...");
}

void loop() {
  float mW = ina219.getPower_mW();
  int phase = digitalRead(PIN_PHASE);

  if (phase != last_phase) {
    if (last_phase != -1 && n > 0) {
      float avg = sum_mW / n;
      if (last_phase == LOW) { last_idle = avg; Serial.print("IDLE phase avg: "); }
      else                   { last_busy = avg; Serial.print("BUSY phase avg: "); }
      Serial.print(avg, 2);
      Serial.print(" mW   (idle=");
      Serial.print(last_idle, 2);
      Serial.print(" busy=");
      Serial.print(last_busy, 2);
      Serial.println(")");
    }
    last_phase = phase;
    sum_mW = 0;
    n = 0;
  }

  sum_mW += mW;
  n++;
  delay(20);
}