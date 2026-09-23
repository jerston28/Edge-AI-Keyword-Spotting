#include <driver/i2s.h>
#include <math.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include "mfcc_test.h"
#include <WiFi.h>
#include <WebServer.h>

const char* WIFI_SSID = "Dummy007";
const char* WIFI_PASS = "12345678";

WebServer server(80);

String g_word = "-";
float g_conf = 0.0f;
int g_latency = 0;
String g_history[5] = {"", "", "", "", ""};
int g_hist_i = 0;

void handleRoot() {
  String html =
    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width, initial-scale=1'>"
    "<title>Keyword Spotter</title>"
    "<style>"
    "body{font-family:sans-serif;background:#111;color:#eee;text-align:center;padding:30px;}"
    "h1{font-size:20px;color:#888;}"
    "#word{font-size:64px;font-weight:bold;color:#4CAF50;margin:20px 0;}"
    "#conf{font-size:22px;color:#ccc;}"
    "#lat{font-size:16px;color:#888;margin-top:10px;}"
    "ul{list-style:none;padding:0;color:#aaa;text-align:left;max-width:300px;margin:30px auto;}"
    "li{padding:6px;border-bottom:1px solid #333;}"
    "</style></head><body>"
    "<h1>ESP32 Live Keyword Spotter</h1>"
    "<div id='word'>-</div>"
    "<div id='conf'>confidence: -</div>"
    "<div id='lat'>latency: -</div>"
    "<h1>Recent detections</h1>"
    "<ul id='hist'></ul>"
    "<script>"
    "async function poll(){"
    "  try{"
    "    let r = await fetch('/status');"
    "    let d = await r.json();"
    "    document.getElementById('word').innerText = d.word;"
    "    document.getElementById('conf').innerText = 'confidence: ' + d.conf;"
    "    document.getElementById('lat').innerText = 'latency: ' + d.latency + ' ms';"
    "    let ul = document.getElementById('hist');"
    "    ul.innerHTML = '';"
    "    d.history.forEach(h => { if(h) { let li = document.createElement('li'); li.innerText = h; ul.appendChild(li); } });"
    "  } catch(e) {}"
    "  setTimeout(poll, 300);"
    "}"
    "poll();"
    "</script></body></html>";
  server.send(200, "text/html", html);
}

void handleStatus() {
  String json = "{";
  json += "\"word\":\"" + g_word + "\",";
  json += "\"conf\":" + String(g_conf, 2) + ",";
  json += "\"latency\":" + String(g_latency) + ",";
  json += "\"history\":[";
  for (int i = 0; i < 5; i++) {
    json += "\"" + g_history[i] + "\"";
    if (i < 4) json += ",";
  }
  json += "]}";
  server.send(200, "application/json", json);
}

#define PIN_SCK 32
#define PIN_WS  25
#define PIN_SD  33
#define PIN_LED 4

#define FRAME_LEN  640
#define FRAME_STEP 320
#define FFT_LEN    1024
#define N_BINS     513
#define N_MEL      40
#define N_MFCC     10
#define N_FRAMES   49
#define RING_SIZE  8192
#define RING_MASK  (RING_SIZE - 1)
#define HOP_FRAMES 16        // run the model every 16 new frames (0.32 s)
#define GAIN       3.0f      // raise if speech looks too quiet
#define THRESH     0.70f     // minimum confidence to count a word
#define ARENA 24576
#define HIST 2

const char* names[8] = {"down", "go", "left", "no", "right", "stop", "up", "yes"};

static int16_t ring[RING_SIZE];
volatile uint32_t total_samples = 0;

static float re[FFT_LEN], im[FFT_LEN];
static float tw_cos[FFT_LEN / 2], tw_sin[FFT_LEN / 2];
static float feat_ring[N_FRAMES][N_MFCC];
static uint32_t next_frame = 0, valid_frames = 0, frames_since_infer = 0;

alignas(16) static uint8_t tensor_arena[ARENA];
static tflite::MicroInterpreter* interp = nullptr;
static TfLiteTensor *input_t, *output_t;

static int last_word = -1, same_count = 0;
static uint32_t last_trigger_ms = 0;

void halt(const char* msg) { Serial.println(msg); while (true) delay(1000); }

// ---------- worker 1: keep listening ----------
void capture_task(void*) {
  static int32_t buf[FRAME_STEP];
  float dc = 0;
  while (true) {
    size_t bytes = 0;
    i2s_read(I2S_NUM_0, buf, sizeof(buf), &bytes, portMAX_DELAY);
    int n = bytes / 4;
    uint32_t t = total_samples;
    for (int i = 0; i < n; i++) {
      float v = (float)(buf[i] >> 8);            // 24-bit mic value
      dc += (v - dc) * 0.002f;                   // track and remove the constant offset
      float y = (v - dc) / 256.0f * GAIN;        // scale to 16-bit range
      if (y > 32767.f) y = 32767.f;
      if (y < -32768.f) y = -32768.f;
      ring[(t + i) & RING_MASK] = (int16_t)y;
    }
    total_samples = t + n;
  }
}

// ---------- MFCC (same maths you already verified) ----------
void fft_init() {
  for (int i = 0; i < FFT_LEN / 2; i++) {
    float a = -2.0f * (float)M_PI * i / FFT_LEN;
    tw_cos[i] = cosf(a); tw_sin[i] = sinf(a);
  }
}

void fft(float* r, float* m) {
  int j = 0;
  for (int i = 1; i < FFT_LEN; i++) {
    int bit = FFT_LEN >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { float t = r[i]; r[i] = r[j]; r[j] = t; t = m[i]; m[i] = m[j]; m[j] = t; }
  }
  for (int len = 2; len <= FFT_LEN; len <<= 1) {
    int half = len >> 1, step = FFT_LEN / len;
    for (int i = 0; i < FFT_LEN; i += len)
      for (int k = 0; k < half; k++) {
        float wr = tw_cos[k * step], wi = tw_sin[k * step];
        int a = i + k, b = a + half;
        float tr = r[b] * wr - m[b] * wi, ti = r[b] * wi + m[b] * wr;
        r[b] = r[a] - tr; m[b] = m[a] - ti; r[a] += tr; m[a] += ti;
      }
  }
}

void compute_frame(uint32_t j, float* out) {
  float mag[N_BINS], logmel[N_MEL];
  uint32_t start = j * FRAME_STEP;
  for (int n = 0; n < FRAME_LEN; n++) {
    re[n] = (ring[(start + n) & RING_MASK] / 32768.0f) * hann_win[n];
    im[n] = 0;
  }
  for (int n = FRAME_LEN; n < FFT_LEN; n++) { re[n] = 0; im[n] = 0; }
  fft(re, im);
  for (int k = 0; k < N_BINS; k++) mag[k] = sqrtf(re[k] * re[k] + im[k] * im[k]);
  for (int m = 0; m < N_MEL; m++) {
    float s = 0;
    for (int k = 0; k < N_BINS; k++) s += mag[k] * mel_matrix[k][m];
    logmel[m] = logf(s + 1e-6f);
  }
  for (int c = 0; c < N_MFCC; c++) {
    float s = 0;
    for (int m = 0; m < N_MEL; m++) s += logmel[m] * dct_mat[m][c];
    out[c] = s;
  }
}

// ---------- setup ----------
void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("\n== Live keyword spotting ==");
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  digitalWrite(PIN_LED, HIGH); delay(1000); digitalWrite(PIN_LED, LOW);

  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate = 16000;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 256;
  cfg.use_apll = false;
  i2s_pin_config_t pins = {};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = PIN_SCK;
  pins.ws_io_num = PIN_WS;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = PIN_SD;
  if (i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL) != ESP_OK) halt("I2S install failed");
  if (i2s_set_pin(I2S_NUM_0, &pins) != ESP_OK) halt("I2S pins failed");

  fft_init();

  const tflite::Model* model = tflite::GetModel(kws_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) halt("Schema version mismatch");
  static tflite::MicroMutableOpResolver<4> resolver;
resolver.AddConv2D();
resolver.AddDepthwiseConv2D();
resolver.AddMean();
resolver.AddFullyConnected();
  static tflite::MicroInterpreter static_interp(model, resolver, tensor_arena, ARENA);
  interp = &static_interp;
  if (interp->AllocateTensors() != kTfLiteOk) halt("AllocateTensors failed");
  input_t = interp->input(0);
  output_t = interp->output(0);
  if ((int)input_t->bytes != N_FRAMES * N_MFCC) halt("Input size mismatch");

  WiFi.begin(WIFI_SSID, WIFI_PASS);
Serial.print("Connecting to Wi-Fi");
while (WiFi.status() != WL_CONNECTED) {
  delay(300);
  Serial.print(".");
}
Serial.println();
Serial.print("Connected. Open this in your browser: http://");
Serial.println(WiFi.localIP());

server.on("/", handleRoot);
server.on("/status", handleStatus);
server.begin();

  xTaskCreatePinnedToCore(capture_task, "capture", 4096, NULL, 5, NULL, 0);
  Serial.println("Listening... say: yes, no, stop, go, up, down, left, right");
}

// ---------- worker 2: fingerprint, model, reaction ----------
void loop() {
  server.handleClient();
  static float hist[HIST][N_CLASSES];
  static int hist_n = 0, hist_i = 0;

  uint32_t avail = total_samples;

  if (next_frame * FRAME_STEP + RING_SIZE - FRAME_LEN < avail) {   // fell too far behind
    next_frame = (avail - FRAME_LEN - 2 * FRAME_STEP) / FRAME_STEP;
    valid_frames = 0; frames_since_infer = 0; hist_n = 0; hist_i = 0;
    Serial.println("(behind: skipped ahead)");
  }

  while (next_frame * FRAME_STEP + FRAME_LEN <= avail) {
    compute_frame(next_frame, feat_ring[next_frame % N_FRAMES]);
    next_frame++;
    if (valid_frames < N_FRAMES) valid_frames++;
    frames_since_infer++;
  }

  if (valid_frames >= N_FRAMES && frames_since_infer >= HOP_FRAMES) {
    frames_since_infer = 0;
    uint32_t t0 = micros();

    for (int f = 0; f < N_FRAMES; f++) {
      uint32_t idx = (next_frame + f) % N_FRAMES;
      for (int c = 0; c < N_MFCC; c++) {
        int q = (int)lroundf(feat_ring[idx][c] / model_in_scale + model_in_zero);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        input_t->data.int8[f * N_MFCC + c] = (int8_t)q;
      }
    }
    if (interp->Invoke() != kTfLiteOk) halt("Invoke failed");
    uint32_t dt = (micros() - t0) / 1000;

    float sc = output_t->params.scale;
    int zp = output_t->params.zero_point;
    float p[N_CLASSES], mx = -1e9f, sum = 0;
    for (int c = 0; c < N_CLASSES; c++) { p[c] = (output_t->data.int8[c] - zp) * sc; if (p[c] > mx) mx = p[c]; }
    for (int c = 0; c < N_CLASSES; c++) { p[c] = expf(p[c] - mx); sum += p[c]; }
    for (int c = 0; c < N_CLASSES; c++) { p[c] /= sum; hist[hist_i][c] = p[c]; }
    hist_i = (hist_i + 1) % HIST;
    if (hist_n < HIST) hist_n++;

    float avg[N_CLASSES];
    int best = 0;
    for (int c = 0; c < N_CLASSES; c++) {
      float s = 0;
      for (int k = 0; k < hist_n; k++) s += hist[k][c];
      avg[c] = s / hist_n;
      if (avg[c] > avg[best]) best = c;
    }

    double sq = 0;
    uint32_t end = total_samples;
    for (int i = 1; i <= 3200; i++) { float v = ring[(end - i) & RING_MASK] / 32768.0f; sq += v * v; }
    float level = sqrt(sq / 3200);

    Serial.printf("level %.3f | %-7s %.2f | %lu ms\n", level, class_names[best], avg[best], (unsigned long)dt);

    bool is_cmd = strcmp(class_names[best], "silence") != 0 && strcmp(class_names[best], "unknown") != 0;
    if (hist_n == HIST && is_cmd && avg[best] >= THRESH && millis() - last_trigger_ms > 1200) {
      last_trigger_ms = millis();
      Serial.printf("*** DETECTED: %s (%.2f) ***\n", class_names[best], avg[best]);

      g_word = class_names[best];
g_conf = avg[best];
g_latency = dt;
g_history[g_hist_i] = g_word + " (" + String(g_conf, 2) + ")";
g_hist_i = (g_hist_i + 1) % 5;
      
      if (strcmp(class_names[best], "yes") == 0) digitalWrite(PIN_LED, HIGH);
      if (strcmp(class_names[best], "no") == 0 || strcmp(class_names[best], "stop") == 0) digitalWrite(PIN_LED, LOW);
    }
  }
  delay(2);
}