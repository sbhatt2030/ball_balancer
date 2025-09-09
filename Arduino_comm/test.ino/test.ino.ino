#include <Servo.h>

Servo ballBalancerServo;
const byte SERVO_PIN = 9;
const byte NEUTRAL_POS = 15;    // Beam horizontal position
const byte MAX_TILT = 15;       // Maximum tilt in either direction
const byte MIN_ANGLE = 0;       // Servo minimum (NEUTRAL - MAX_TILT)
const byte MAX_ANGLE = 30;      // Servo maximum (NEUTRAL + MAX_TILT)

// Sine wave parameters
float wavePhase = 0.0;
const float WAVE_SPEED = 0.02;  // Phase increment per loop (controls frequency)
const int UPDATE_DELAY = 25;    // milliseconds between updates (40Hz)

bool testRunning = false;
unsigned long lastUpdate = 0;
int currentAngle = NEUTRAL_POS;

void setup() {
  Serial.begin(9600);
  ballBalancerServo.attach(SERVO_PIN);
  ballBalancerServo.write(NEUTRAL_POS);
  
  Serial.println("=== Ball Balancer Servo Test ===");
  Serial.println("Range: 0° to 30° (±15° from center)");
  Serial.println("Commands:");
  Serial.println("  'START' - Begin sine wave test");
  Serial.println("  'STOP'  - End test and center beam");
  Serial.println("  '0-30'  - Set specific angle");
  Serial.println("Ready for commands...");
}

void loop() {
  handleSerialInput();
  
  if (testRunning && (millis() - lastUpdate >= UPDATE_DELAY)) {
    generateSineWave();
    lastUpdate = millis();
  }
}

void handleSerialInput() {
  if (!Serial.available()) return;
  
  String input = Serial.readStringUntil('\n');
  input.trim();
  input.toUpperCase();
  
  if (input == "START") {
    startSineTest();
  }
  else if (input == "STOP") {
    stopSineTest();
  }
  else {
    // Try to parse as angle
    int angle = input.toInt();
    if (angle >= MIN_ANGLE && angle <= MAX_ANGLE) {
      setManualAngle(angle);
    } else {
      Serial.println("Invalid command. Use START/STOP or angle 0-30");
    }
  }
}

void startSineTest() {
  testRunning = true;
  wavePhase = 0.0;
  lastUpdate = millis();
  Serial.println("Sine wave test ACTIVE - beam will oscillate ±15°");
  Serial.println("Send 'STOP' to halt test");
}

void stopSineTest() {
  testRunning = false;
  currentAngle = NEUTRAL_POS;
  ballBalancerServo.write(currentAngle);
  Serial.println("Test STOPPED - beam returned to center");
}

void setManualAngle(int angle) {
  testRunning = false;
  currentAngle = constrain(angle, MIN_ANGLE, MAX_ANGLE);
  ballBalancerServo.write(currentAngle);
  
  int tiltFromCenter = currentAngle - NEUTRAL_POS;
  Serial.print("Manual position: ");
  Serial.print(currentAngle);
  Serial.print("° (");
  Serial.print(tiltFromCenter > 0 ? "+" : "");
  Serial.print(tiltFromCenter);
  Serial.println("° from center)");
}

void generateSineWave() {
  // Generate smooth sine wave
  float sineOutput = sin(wavePhase);
  
  // Map sine (-1 to +1) to servo angle range
  currentAngle = NEUTRAL_POS + (int)(MAX_TILT * sineOutput);
  
  // Safety bounds check
  currentAngle = constrain(currentAngle, MIN_ANGLE, MAX_ANGLE);
  
  // Move servo
  ballBalancerServo.write(currentAngle);
  
  // Advance phase for next cycle
  wavePhase += WAVE_SPEED;
  if (wavePhase >= TWO_PI) {
    wavePhase -= TWO_PI;
    Serial.println("Completed one full oscillation cycle");
  }
  
  // Optional: Print angle every 20 updates (about once per second at 40Hz)
  static int printCounter = 0;
  if (++printCounter >= 20) {
    int tilt = currentAngle - NEUTRAL_POS;
    Serial.print("Angle: ");
    Serial.print(currentAngle);
    Serial.print("° (tilt: ");
    Serial.print(tilt > 0 ? "+" : "");
    Serial.print(tilt);
    Serial.println("°)");
    printCounter = 0;
  }
}