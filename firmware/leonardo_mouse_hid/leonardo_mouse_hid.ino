/*
  Leonardo Mouse HID 固件
  Board: Arduino Leonardo / Pro Micro(ATmega32U4) / 兼容板
  Baud: 115200

  串口协议，与 src/leonardo_driver.py 对应：
    Packet: [0xAA, cmd, dx, dy, checksum]
    checksum = (cmd + dx + dy) & 0xFF
    dx/dy 使用 int8_t，相对移动范围 -127..127

  cmd:
    0x01 MOVE
    0x02 LEFT PRESS
    0x03 LEFT RELEASE
    0x04 LEFT CLICK
    0x05 MOVE + PRESS
    0x06 MOVE + RELEASE
    0xFF HEARTBEAT，返回 0xBB
*/

#include <Mouse.h>

static const uint8_t HEADER = 0xAA;
static const uint8_t RESP_OK = 0xBB;
static const uint8_t RESP_BAD = 0xEE;

static const uint8_t CMD_MOVE = 0x01;
static const uint8_t CMD_PRESS = 0x02;
static const uint8_t CMD_RELEASE = 0x03;
static const uint8_t CMD_CLICK = 0x04;
static const uint8_t CMD_MOVE_PRESS = 0x05;
static const uint8_t CMD_MOVE_RELEASE = 0x06;
static const uint8_t CMD_HEARTBEAT = 0xFF;

static const unsigned long SERIAL_BAUD = 115200;
static const unsigned long FAILSAFE_RELEASE_MS = 2000;
static const int MAX_STEP = 127;

bool leftPressed = false;
unsigned long lastCommandMs = 0;

int8_t toSigned8(uint8_t v) {
  return (int8_t)v;
}

void releaseLeftIfNeeded() {
  if (leftPressed) {
    Mouse.release(MOUSE_LEFT);
    leftPressed = false;
  }
}

void doMove(int8_t dx, int8_t dy) {
  int x = constrain((int)dx, -MAX_STEP, MAX_STEP);
  int y = constrain((int)dy, -MAX_STEP, MAX_STEP);
  if (x != 0 || y != 0) {
    Mouse.move(x, y, 0);
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  Mouse.begin();
  lastCommandMs = millis();
}

void loop() {
  if (millis() - lastCommandMs > FAILSAFE_RELEASE_MS) {
    releaseLeftIfNeeded();
  }

  if (Serial.available() < 5) {
    return;
  }

  int first = Serial.read();
  if (first != HEADER) {
    return;
  }

  while (Serial.available() < 4) {
    delayMicroseconds(200);
  }

  uint8_t cmd = (uint8_t)Serial.read();
  uint8_t dxu = (uint8_t)Serial.read();
  uint8_t dyu = (uint8_t)Serial.read();
  uint8_t checksum = (uint8_t)Serial.read();
  uint8_t expected = (uint8_t)((cmd + dxu + dyu) & 0xFF);

  if (checksum != expected) {
    Serial.write(RESP_BAD);
    return;
  }

  lastCommandMs = millis();
  int8_t dx = toSigned8(dxu);
  int8_t dy = toSigned8(dyu);

  switch (cmd) {
    case CMD_MOVE:
      doMove(dx, dy);
      break;

    case CMD_PRESS:
      Mouse.press(MOUSE_LEFT);
      leftPressed = true;
      break;

    case CMD_RELEASE:
      releaseLeftIfNeeded();
      break;

    case CMD_CLICK:
      Mouse.click(MOUSE_LEFT);
      leftPressed = false;
      break;

    case CMD_MOVE_PRESS:
      doMove(dx, dy);
      Mouse.press(MOUSE_LEFT);
      leftPressed = true;
      break;

    case CMD_MOVE_RELEASE:
      doMove(dx, dy);
      releaseLeftIfNeeded();
      break;

    case CMD_HEARTBEAT:
      Serial.write(RESP_OK);
      break;

    default:
      Serial.write(RESP_BAD);
      break;
  }
}
