#include <Arduino.h>
#include <Stepper.h>

#define STEPS_PER_ROT 658560

Stepper s_alt(48, 4, 5, 6, 7);
Stepper s_azi(48, 8, 9, 10, 11);

uint32_t steps_alt = 0, steps_azi = 0;

float steps_to_deg(uint32_t steps) {
    return ((float)steps) * 360.0f / ((float)STEPS_PER_ROT);
}

uint32_t deg_to_steps(float deg) {
    return deg * ((float)STEPS_PER_ROT) / 360.0f;
}

void setup() {
    Serial.begin(9600, SERIAL_8N1);
    Serial.setTimeout(100);
    s_alt.setSpeed(1400);
    s_azi.setSpeed(1400);
    DDRD |= B11110000;
    DDRB |= B00001111;
}

void loop() {
    if (Serial.available() > 0) {
        char cmd = Serial.read();
        float arg1 = Serial.parseFloat();
        float arg2 = Serial.parseFloat();
        Serial.read();
        if (cmd == 's') {
            steps_alt = deg_to_steps(arg1);
            steps_azi = deg_to_steps(arg2);
            Serial.println("done");
        } else if (cmd == 'm') {
            bool c1, c2;
            uint32_t set_steps_alt = deg_to_steps(arg1);
            uint32_t set_steps_azi = deg_to_steps(arg2);
            int8_t dir1 = set_steps_alt >= steps_alt ? 1 : -1;
            int8_t dir2 = set_steps_azi >= steps_azi ? 1 : -1;
            while (
                (c1 = (steps_alt != set_steps_alt)) ||
                (c2 = (steps_azi != set_steps_azi))
            ) {
                if (c1) {
                    s_alt.step(dir1);
                    steps_alt += dir1;
                }
                if (c2) {
                    s_azi.step(dir2);
                    steps_azi += dir2;
                }
            }
            Serial.println("done");
        } else if (cmd == 'c') {
            Serial.println(steps_to_deg(steps_alt));
            Serial.println(steps_to_deg(steps_azi));
        } else {
            Serial.println("error");
        }
    }
}
