#include <math.h>
#include "mfcc_test.h"

#define FRAME_LEN  640
#define FRAME_STEP 320
#define FFT_LEN    1024
#define N_BINS     513
#define N_MEL      40
#define N_MFCC     10
#define N_FRAMES   49

static float re[FFT_LEN], im[FFT_LEN];
static float tw_cos[FFT_LEN / 2], tw_sin[FFT_LEN / 2];
static float feat[N_FRAMES][N_MFCC];

void fft_init() {
  for (int i = 0; i < FFT_LEN / 2; i++) {
    float a = -2.0f * (float)M_PI * i / FFT_LEN;
    tw_cos[i] = cosf(a);
    tw_sin[i] = sinf(a);
  }
}

void fft(float* r, float* m) {
  int j = 0;
  for (int i = 1; i < FFT_LEN; i++) {           // reorder the samples
    int bit = FFT_LEN >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float t = r[i]; r[i] = r[j]; r[j] = t;
      t = m[i]; m[i] = m[j]; m[j] = t;
    }
  }
  for (int len = 2; len <= FFT_LEN; len <<= 1) { // combine
    int half = len >> 1, step = FFT_LEN / len;
    for (int i = 0; i < FFT_LEN; i += len) {
      for (int k = 0; k < half; k++) {
        float wr = tw_cos[k * step], wi = tw_sin[k * step];
        int a = i + k, b = a + half;
        float tr = r[b] * wr - m[b] * wi;
        float ti = r[b] * wi + m[b] * wr;
        r[b] = r[a] - tr; m[b] = m[a] - ti;
        r[a] += tr;       m[a] += ti;
      }
    }
  }
}

void compute_mfcc(const int16_t* pcm) {
  float mag[N_BINS], logmel[N_MEL];
  for (int f = 0; f < N_FRAMES; f++) {
    const int16_t* p = pcm + f * FRAME_STEP;
    for (int n = 0; n < FRAME_LEN; n++) { re[n] = (p[n] / 32768.0f) * hann_win[n]; im[n] = 0; }
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
      feat[f][c] = s;
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("\n== On-chip MFCC test ==");
  fft_init();

  uint32_t t0 = micros();
  compute_mfcc(test_pcm);
  uint32_t dt = micros() - t0;

  float max_err = 0;
  int off_by_one = 0, off_more = 0;
  for (int f = 0; f < N_FRAMES; f++) {
    for (int c = 0; c < N_MFCC; c++) {
      float d = fabsf(feat[f][c] - ref_mfcc[f][c]);
      if (d > max_err) max_err = d;
      int q = (int)lroundf(feat[f][c] / in_scale + in_zero);
      if (q > 127) q = 127;
      if (q < -128) q = -128;
      int diff = abs(q - (int)ref_q[f][c]);
      if (diff == 1) off_by_one++;
      if (diff > 1) off_more++;
    }
  }
  Serial.printf("MFCC time for 1 second of audio: %.1f ms\n", dt / 1000.0f);
  Serial.printf("Max float difference vs Colab: %.5f\n", max_err);
  Serial.printf("int8 values off by 1: %d / 490,  off by more than 1: %d\n", off_by_one, off_more);
  Serial.println("Frame 0, chip vs Colab:");
  for (int c = 0; c < N_MFCC; c++) Serial.printf("  %8.4f  %8.4f\n", feat[0][c], ref_mfcc[0][c]);
}

void loop() {}