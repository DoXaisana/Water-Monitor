#include <Arduino.h>
#include <LiquidCrystal_I2C.h>

int ledGreen = 8;
int ledBlue = 9;
int ledRed = 10;
float waterLevel = 0;
int waterLevelPin = A0;

LiquidCrystal_I2C lcd(0x27, 16, 2);

void setup() {
  pinMode(ledGreen, OUTPUT);
  pinMode(ledBlue, OUTPUT);
  pinMode(ledRed, OUTPUT);
  pinMode(waterLevelPin, INPUT);
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Water Level:");
}

void loop() {
  waterLevel = analogRead(waterLevelPin);
  waterLevel = map(waterLevel, 0, 520, 0, 100);
  lcd.setCursor(0, 1);
  lcd.print(waterLevel);
  lcd.print("%");
  if (waterLevel < 30) {
    digitalWrite(ledGreen, LOW);
    digitalWrite(ledBlue, LOW);
    digitalWrite(ledRed, HIGH);
  } else if (waterLevel < 70) {
    digitalWrite(ledGreen, LOW);
    digitalWrite(ledBlue, HIGH);
    digitalWrite(ledRed, LOW);
  } else {
    digitalWrite(ledGreen, HIGH);
    digitalWrite(ledBlue, LOW);
    digitalWrite(ledRed, LOW);
  }
  delay(1000);
}