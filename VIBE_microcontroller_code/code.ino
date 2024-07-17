/**
 * @file VIBE_Microcontroller_code.ino
 * @brief Firmware running in Seeed XIAO BLE Sense - nRF52840 to use the downloaded arduino library from 
 * Edge impulse pltform for Sound classification. 
 * @version 1.0
 * @date 2024-07-02
 * @Velmurugan S [EE21s131]
 * 
 * This code is designed to classify sounds using a pre-trained model and display the 
 * results on a Display and controlling a piezo motor based on the classification results.
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include "Adafruit_GFX.h"
#include "Adafruit_GC9A01A.h"
#include "SPI.h"
#define TFT_DC 4
#define TFT_CS 5
Adafruit_GC9A01A tft(TFT_CS, TFT_DC);
// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0

unsigned long startTime;
unsigned long stopTime;
unsigned long elapsedTime;
bool timerRunning = false;

/**
 * Define the number of slices per model window. E.g. a model window of 1000 ms
 * with slices per model window set to 4. Results in a slice size of 250 ms.
 * For more info: https://docs.edgeimpulse.com/docs/continuous-audio-sampling
 */
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 4

/* Includes ---------------------------------------------------------------- */
#include <PDM.h>
#include <Algorithm_Kanagarag_v2_inferencing.h>
//#include <my_model_inferencing.h>
//#include <Algorithm_15thsep_silence_inferencing.h>
//#include <test_model_inferencing.h>
//
int motorPin = 1;
/** Audio buffers, pointers and selectors */
typedef struct {
    signed short *buffers[2];
    unsigned char buf_select;
    unsigned char buf_ready;
    unsigned int buf_count;
    unsigned int n_samples;
} inference_t;

static inference_t inference;
static bool record_ready = false;
static signed short *sampleBuffer;
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);

/**
 * @brief      Arduino setup function
 */
void setup()
{
    // put your setup code here, to run once:
    Serial.begin(115200);
    tft.begin();
    tft.setRotation(4);
    pinMode(motorPin, OUTPUT);
    delay(1000);
    Serial.println("Edge Impulse Inferencing Demo");
   
   
    // summary of inferencing settings (from model_metadata.h)
    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) /
                                            sizeof(ei_classifier_inferencing_categories[0]));

    run_classifier_init();
    if (microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE) == false) {
        ei_printf("ERR: Failed to setup audio sampling\r\n");
        return;
    }
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop()
{
    bool m = microphone_inference_record();
    if (!m) {
        //ei_printf("ERR: Failed to record audio...\n");
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};

    EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, debug_nn);
    
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    if (++print_results >= (EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)) {
        float p = 0;
        int idx = 0;
       
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf("    %s: %.1f\n", result.classification[ix].label,
                      result.classification[ix].value); 
            // This printf line returns the probability of all classes for the processed 1-second sample
            if (result.classification[ix].value > p) {
              p = result.classification[ix].value ;
              idx = ix;
            }
        }

#if EI_CLASSIFIER_HAS_ANOMALY == 1
        ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif

        float probabilities[6]; // Array to hold probabilities for each class
        const char* labels[] = {"Alarm", "Baby Cry", "Door knock", "Vehicle horn", "Spoken name", "No sound"}; // Labels for each class

        // Populate the probabilities array with your classification results
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            probabilities[ix] = result.classification[ix].value; // Assuming result.classification[ix].value holds the probability
        }

        // Find the index of the class with the highest probability
        int maxIndex = 0;
        for (int i = 1; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
            if (probabilities[i] > probabilities[maxIndex]) {
                maxIndex = i;
            }
        }

        // Display the message based on the highest probability class
        displayMessage(labels[maxIndex], true, probabilities[maxIndex]);

        print_results = 0;
    }
}





/**
 * @brief      Printf function uses vsnprintf and output using Arduino Serial
 *
 * @param[in]  format     Variable argument list
 */
void ei_printf(const char *format, ...) {
    static char print_buf[1024] = { 0 };

    va_list args;
    va_start(args, format);
    int r = vsnprintf(print_buf, sizeof(print_buf), format, args);
    va_end(args);

    if (r > 0) {
        Serial.write(print_buf);
    }
}

/**
 * @brief      PDM buffer full callback
 *             Get data and call audio thread callback
 */
static void pdm_data_ready_inference_callback(void)
{
    int bytesAvailable = PDM.available();

    // read into the sample buffer
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);
    
    if (record_ready == true) {
        for (int i = 0; i<bytesRead>> 1; i++) {
            inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];
            // Print the current buffer value every 100 samples to avoid overwhelming the serial output
            
            if (inference.buf_count >= inference.n_samples) {
                inference.buf_select ^= 1;
                inference.buf_count = 0;
                inference.buf_ready = 1;
            }
        }
    }
}

/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
static bool microphone_inference_start(uint32_t n_samples)
{
    
    inference.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[0] == NULL) {
        return false;
    }

    inference.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[0] == NULL) {
        free(inference.buffers[0]);
        return false;
    }

    sampleBuffer = (signed short *)malloc((n_samples >> 1) * sizeof(signed short));

    if (sampleBuffer == NULL) {
        free(inference.buffers[0]);
        free(inference.buffers[1]);
        return false;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    // configure the data receive callback
    PDM.onReceive(&pdm_data_ready_inference_callback);

    // optionally set the gain, defaults to 20
    PDM.setGain(80);

    PDM.setBufferSize((n_samples >> 1) * sizeof(int16_t));
    


    // initialize PDM with:
    // - one channel (mono mode)
    // - a 16 kHz sample rate
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
    }

    record_ready = true;

    return true;
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool microphone_inference_record(void)
{
    bool ret = true;

    if (inference.buf_ready == 1) {
//        ei_printf(
//            "Error sample buffer overrun. Decrease the number of slices per model window "
//            "(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)\n");
        ret = false;
    }

    while (inference.buf_ready == 0) {
        delay(10);
    }

    inference.buf_ready = 0;

    return ret;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);

    return 0;
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void)
{
    PDM.end();
    free(inference.buffers[0]);
    free(inference.buffers[1]);
}

void displayMessage(const char* message, bool motorOn, float p) {
    tft.fillScreen(GC9A01A_WHITE);
    tft.setCursor(0, 4);
    tft.setTextColor(GC9A01A_BLACK);
    tft.setTextSize(3);
    tft.setCursor(40, 100);
    tft.print(message);
    tft.print(" ");
    tft.print(p * 100, 2);
    tft.println("%");

    if (motorOn) {
        digitalWrite(motorPin, HIGH);
        delay(300);
        digitalWrite(motorPin, LOW);
    } else {
        digitalWrite(motorPin, LOW);
    }
    
    delay(1000);
}
