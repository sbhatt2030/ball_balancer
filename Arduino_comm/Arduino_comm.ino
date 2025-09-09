#include <Servo.h>

Servo beamServo;
const int servoPin = 9;
int angle = 15; // Beam at rest

void setup() {
  Serial.begin(9600);
  beamServo.attach(servoPin);
  beamServo.write(angle);
}

void loop() {
  if (Serial.available()) {
  angle = Serial.read();  // read raw byte;
  beamServo.write(angle);
  Serial.print("Angle received: ");
  // Serial.println(angle);
  }
}

