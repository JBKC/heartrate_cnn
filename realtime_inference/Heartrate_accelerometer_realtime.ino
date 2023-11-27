// Arduino Uno code for extracting PPG and accelerometer sensor readings
// Used as input to "cnn_realtime_inference_continuous"

#include <Wire.h>
#include <Adafruit_MMA8451.h>
#include <Adafruit_Sensor.h>
Adafruit_MMA8451 mma = Adafruit_MMA8451();

// intialisation
const int PPGinput = A0;              // Change input pin as necessary
unsigned long previousMillis = 0;     // Variable to store the last time the timer was updated
unsigned long segmentTime = 2000;     // New segment every 2000ms
bool initialSegment = true;           // Flag to determine the initial segment
int segmentCounter = 0;               // Counter to keep track of resets

void setup(void) {
  Serial.begin(115200);
  Serial.println("Output started");
  // error flagging
  if (! mma.begin()) {
    Serial.println("Couldnt start");
    while (1);
  }  
  mma.setRange(MMA8451_RANGE_2_G);  // Set range of accelerometer
}

void loop() {
  unsigned long currentTime = millis();

  if (currentTime - previousMillis >= segmentTime) {
    // if time exceeds the segment time
    
    // Reset the timer
    previousMillis = currentTime;
    
    // Add 1 to the segment counter
    segmentCounter++;
    Serial.print("SEGMENT,");
    Serial.println(segmentCounter);
  }

  // Overall timer
  Serial.print("TIME,");
  Serial.println(currentTime);

  // Extract accelerometer data
  mma.read();
  sensors_event_t event; 
  mma.getEvent(&event);

  // // Format and send accelerometer data
  Serial.print("ACCEL,");
  Serial.print(event.acceleration.x); Serial.print(",");
  Serial.print(event.acceleration.y); Serial.print(",");
  Serial.print(event.acceleration.z); Serial.print(",");
  Serial.print(segmentCounter); 
  Serial.println();

  // Extract PPG data
  float inputSignal = analogRead(PPGinput);

  // Format and send PPG data
  Serial.print("PPG,"); 
  Serial.print(inputSignal); Serial.print(",");
  Serial.print(segmentCounter);
  Serial.println();
}
