#include <Servo.h>

Servo beamServo;
const int servoPin = 9;

// Servo limits and defaults
const int MIN_ANGLE = 0;
const int MAX_ANGLE = 30;
const int NEUTRAL_ANGLE = 15;

int currentAngle = NEUTRAL_ANGLE;
unsigned long lastCommandTime = 0;
const unsigned long TIMEOUT_MS = 5000; // 5 second timeout

void setup() {
  Serial.begin(9600);
  beamServo.attach(servoPin);
  
  // Move to neutral position on startup
  beamServo.write(NEUTRAL_ANGLE);
  currentAngle = NEUTRAL_ANGLE;
  
  // Send ready signal
  Serial.println("SERVO_READY");
  Serial.print("NEUTRAL_ANGLE:");
  Serial.println(NEUTRAL_ANGLE);
  Serial.print("MIN_MAX:");
  Serial.print(MIN_ANGLE);
  Serial.print(",");
  Serial.println(MAX_ANGLE);
}

void loop() {
  // Check for incoming serial data
  if (Serial.available()) {
    int receivedAngle = Serial.read();  // Read raw byte (0-255)
    
    // Validate angle range
    if (receivedAngle >= MIN_ANGLE && receivedAngle <= MAX_ANGLE) {
      currentAngle = receivedAngle;
      beamServo.write(currentAngle);
      lastCommandTime = millis();
      
      // Send acknowledgment
      Serial.print("ANGLE_SET:");
      Serial.println(currentAngle);
    } else {
      // Send error for invalid angle
      Serial.print("ERROR:INVALID_ANGLE:");
      Serial.println(receivedAngle);
    }
  }
  
  // Safety timeout - return to neutral if no commands received
  if (millis() - lastCommandTime > TIMEOUT_MS && currentAngle != NEUTRAL_ANGLE) {
    currentAngle = NEUTRAL_ANGLE;
    beamServo.write(currentAngle);
    Serial.println("TIMEOUT:RETURNING_TO_NEUTRAL");
    lastCommandTime = millis(); // Reset timer to avoid repeated messages
  }
  
  // Small delay to prevent overwhelming serial
  delay(10);
}

// Optional: Handle special commands via serial string input
void serialEvent() {
  // This function is called automatically when serial data is available
  // Can be used for more complex command parsing if needed
}