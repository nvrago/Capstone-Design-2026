/*
 * ClearCoreModbusTest.ino
 *
 * Scan-to-Mill comms test — ClearCore side.
 *
 * ClearCore runs a Modbus TCP server on port 502. The register map follows
 * the Teknic Modbus HMI reference style: a single packed CLIENT_INFC struct
 * is the source of truth, and Modbus register addresses are computed from
 * offsetof(field)/2. The wire protocol is therefore self-documenting via
 * the struct definition — add a field, recompile, the register map updates.
 *
 * The motor on M-0 runs back and forth between +TEST_STROKE_STEPS and
 * -TEST_STROKE_STEPS as "Sequence 1". This auto-starts at boot (so you can
 * verify the motor even without the Pi connected) and can be kicked off
 * again by the Pi sending CCMD_RUN_1. CCMD_STOP halts immediately via
 * MoveStopAbrupt().
 *
 * Register map (CLIENT_INFC):
 *
 *   Command channel (Pi writes, ClearCore reads):
 *     word 0-1   acc           uint32_t  steps/sec^2
 *     word 2-3   vel           int32_t   steps/sec (magnitude)
 *     word 4-5   target_posn   int32_t   steps, absolute
 *     word 6     cmd           uint16_t  CCMTR_CMD
 *
 *   Status channel (ClearCore writes, Pi reads):
 *     word 7-8   cur_posn      int32_t   steps
 *     word 9     status        uint16_t  CCMTR_STATUS bitfield
 *     word 10    msg_cnt       uint16_t  bumped each setMsg() call
 *     word 11-42 msg           char[64]  ClearCore-side debug text
 *     word 43    state         uint16_t  CCMTR_STATE
 *
 * Endianness note:
 *   ClearCore is ARM Cortex-M4 (little-endian), and the Modbus handler
 *   accesses the struct via reinterpret_cast<uint16_t*>. So 32-bit fields
 *   appear on the wire low-word-first. The Pi side (UI.py) matches this
 *   with pack_i32() / unpack_i32() helpers. If you change either side,
 *   change both.
 *
 * About "interrupts":
 *   Modbus TCP is polled — the Arduino Ethernet wrapper services LwIP
 *   internally on each call, so we just keep looping through serviceModbus().
 *   The loop runs <1 ms with nothing else going on, so a STOP from the
 *   panel reaches MoveStopAbrupt() within a couple of milliseconds of the
 *   TCP packet arriving.
 *
 *   A true hardware-interrupt path is available for a physical panic button
 *   on DI-6 — see the attachInterrupt block at the bottom of setup() and
 *   panicButtonISR() at the bottom of the file. The ISR only sets a flag;
 *   serviceMotion() catches the flag at the top of the next iteration and
 *   does the actual halt.
 */

#include "ClearCore.h"
#include <SPI.h>
#include <Ethernet.h>
#include <stddef.h>   // offsetof
#include <string.h>

using namespace ClearCore;

// ── Network config ────────────────────────────────────────────────────────
#define USE_DHCP false

// Must match MODBUS_DEFAULT_HOST in the Pi's UI.py
IPAddress STATIC_IP (192, 168, 1, 20);
IPAddress NETMASK   (255, 255, 255, 0);
IPAddress GATEWAY   (192, 168, 1, 1);
IPAddress DNS_IP    (192, 168, 1, 1);

// ClearCore's Arduino wrapper ignores the MAC bytes you pass and uses the
// factory-programmed MAC from the chip. We still need to supply *something*
// to satisfy the Ethernet.begin() signature.
byte MAC_PLACEHOLDER[] = { 0x24, 0x15, 0x10, 0x00, 0x00, 0x00 };

#define MODBUS_TCP_PORT 502
#define MODBUS_UNIT_ID  1

// ── Motion config ─────────────────────────────────────────────────────────
#define motor             ConnectorM0
#define TEST_STROKE_STEPS 5000      // (unused now, kept for reference)
#define TEST_VEL_SPS      10000     // steps/sec — fast (rapid)
#define TEST_ACCEL_SPSPS  100000    // steps/sec^2

// Scan/home motion (slow, deliberate)
#define SCAN_VEL_SPS      2000      // steps/sec — slow scan/home rate
#define SCAN_ACCEL_SPSPS  20000     // steps/sec^2
#define MAX_TRAVEL_STEPS  1000000   // big enough to guarantee hitting a limit

// Limit switches
#define homeLimit         ConnectorIO3   // "home" side limit switch
#define farLimit          ConnectorIO2   // "far"  side limit switch
#define estopInput        ConnectorIO1   // emergency stop button

// Discrete-capture scan parameters
#define SCAN_STEP_STEPS    10000    // steps advanced per capture point
#define SCAN_SETTLE_MS     150      // ring-down time after motion stops
#define SCAN_MAX_POINTS    64       // safety cap, prevents runaway

// Active level: most mechanical NC switches read LOW when pressed.
// If yours read HIGH when triggered, change this to true.
#define LIMIT_ACTIVE_HIGH true

static inline bool homeLimitHit() {
    return LIMIT_ACTIVE_HIGH ? homeLimit.State() : !homeLimit.State();
}
static inline bool farLimitHit() {
    return LIMIT_ACTIVE_HIGH ? farLimit.State()  : !farLimit.State();
}

static inline bool estopEngaged() {
    return LIMIT_ACTIVE_HIGH ? estopInput.State() : !estopInput.State();
}
// ── Command enum (mirrors Teknic reference shared.h) ──────────────────────
typedef enum {
    CCMD_NONE       = 0,
    CCMD_ENAB_MTRS  = 1,    // enable motor drive
    CCMD_DISAB_MTRS = 2,    // disable motor drive
    CCMD_SET_ZERO   = 3,    // zero current position (not implemented in test)
    CCMD_MOVE       = 4,    // use acc/vel/target_posn to move absolute
    CCMD_STOP       = 5,    // abrupt stop, keep drive enabled
    CCMD_RUN_1      = 6,    // start/restart sequence 1 (back-and-forth)
    CCMD_ACK        = 7,
    CCMD_NACK       = 8
    CCMD_NEXT_POINT  = 9    // Pi ack: capture done, advance to next point

} CCMTR_CMD;

// ── Controller state enum ─────────────────────────────────────────────────
typedef enum {
    CST_INIT     = 0,
    CST_IDLE     = 1,
    CST_ENABLED  = 2,
    CST_RUNNING  = 3,
    CST_STOPPED  = 4,
    CST_FAULT    = 5,
    CST_UNKNOWN  = 6
} CCMTR_STATE;

// ── Status bitfield ───────────────────────────────────────────────────────
// Bit layout matches what UI.py parses. First-declared bit is LSB under
// GCC/little-endian ARM, which is the ClearCore toolchain.
#pragma pack(push, 1)
struct CCMTR_STATUS {
    struct bits {
        uint16_t drvs_enabled : 1;  // bit 0 - drive enabled ("ready")
        uint16_t moving       : 1;  // bit 1 - motion in progress
        uint16_t homed        : 1;  // bit 2 - homing complete
        uint16_t fault        : 1;  // bit 3 - fault latched
        uint16_t estop        : 1;  // bit 4 - estop latched
        uint16_t scanning     : 1;  // bit 5 - sequence 1 running
        uint16_t hlfb         : 1;  // bit 6 - ClearPath HLFB OK (unused here)
        uint16_t at_capture   : 1;   // bit 7 - stopped & settled, ready for capture
        uint16_t unused       : 9;
    };
    union {
        struct bits b;
        uint16_t    reg16;
    };
    CCMTR_STATUS() : reg16(0) { }
};
#pragma pack(pop)

// ── Shared register block ─────────────────────────────────────────────────
#pragma pack(push, 1)
struct CLIENT_INFC {
    // Command channel — Pi writes, ClearCore reads
    uint32_t     acc;           // word 0-1   steps/sec^2
    int32_t      vel;           // word 2-3   steps/sec
    int32_t      target_posn;   // word 4-5   steps (absolute)
    uint16_t     cmd;           // word 6     CCMTR_CMD

    // Status channel — ClearCore writes, Pi reads
    int32_t      cur_posn;      // word 7-8   steps
    CCMTR_STATUS status;        // word 9     bitfield
    uint16_t     msg_cnt;       // word 10    bumped per setMsg()
    char         msg[64];       // word 11-42 ClearCore debug text
    uint16_t     state;         // word 43    CCMTR_STATE
};
#pragma pack(pop)

const uint16_t NUM_REGS = sizeof(CLIENT_INFC) / 2;
uint32_t rearmStartMs = 0;

static uint32_t settleStartMs   = 0;
static uint16_t capturePointIdx = 0;

// Compile-time word-offset helper — matches offsetof/2 in Teknic reference
#define W_ADDR(field) ((uint16_t)(offsetof(CLIENT_INFC, field) / 2))

// ── Modbus wire constants ─────────────────────────────────────────────────
#define FC_READ_HOLDING       0x03
#define FC_WRITE_SINGLE_REG   0x06
#define FC_WRITE_MULTIPLE_REG 0x10

#define EX_ILLEGAL_FUNCTION   0x01
#define EX_ILLEGAL_ADDRESS    0x02

// ── Globals ───────────────────────────────────────────────────────────────
EthernetServer modbusServer(MODBUS_TCP_PORT);
EthernetClient modbusClient;

CLIENT_INFC regs;  // the one and only shared register block

enum MotionState {
    MS_IDLE,
    MS_REARM_WAIT,    // waiting for drive to come up after enable
    MS_HOMING,        // moving toward homeLimit, waiting for it to trigger
    MS_SCAN_STEP,        // commanded a step-sized move, waiting for completion
    MS_SETTLE,           // motion stopped, timing the ring-down
    MS_AT_CAPTURE,       // settled, waiting for Pi CCMD_NEXT_POINT
    MS_RETURNING,     // slow return from far back to homeLimit
    MS_STOPPED,
    MS_DISABLED
};

volatile MotionState motionState   = MS_IDLE;
volatile bool        stopRequested = false;  // set by panic button ISR

char dbg[96];  // scratch for Serial prints

// ── Forward decls ─────────────────────────────────────────────────────────
void setupMotor();
void setupEthernet();
void serviceModbus();
void serviceMotion();
void handleCommand(uint16_t cmd);
void updateStatusRegisters();
void startSequence1();
void sendModbusException(uint8_t *mbapHdr, uint8_t fc, uint8_t exCode);
void setMsg(const char *text);
void panicButtonISR();

static inline uint16_t readU16(const uint8_t *buf) {
    return ((uint16_t)buf[0] << 8) | buf[1];
}
static inline void writeU16(uint8_t *buf, uint16_t val) {
    buf[0] = (val >> 8) & 0xFF;
    buf[1] = val & 0xFF;
}

// ── setup() / loop() ──────────────────────────────────────────────────────
void setup() {
    Serial.begin(9600);
    uint32_t t0 = millis();
    while (!Serial && (millis() - t0) < 3000) { continue; }
    Serial.println("[BOOT] Scan-to-Mill ClearCore Modbus TCP test");
    sprintf(dbg, "[BOOT] CLIENT_INFC = %u bytes = %u regs, cmd@w%u",
            (unsigned)sizeof(CLIENT_INFC), NUM_REGS, (unsigned)W_ADDR(cmd));
    Serial.println(dbg);

    // Initialize the shared block
    memset(&regs, 0, sizeof(regs));
    regs.acc   = TEST_ACCEL_SPSPS;
    regs.vel   = TEST_VEL_SPS;
    regs.cmd   = CCMD_NONE;
    regs.state = CST_INIT;

    setupMotor();
    setupEthernet();

    modbusServer.begin();
    sprintf(dbg, "[NET] Modbus TCP listening on :%d unit %d",
            MODBUS_TCP_PORT, MODBUS_UNIT_ID);
    Serial.println(dbg);

    // Auto-start sequence 1 so the motor moves without waiting for the Pi.
    // If you see motion here but the Pi can't stop it, it's a comms problem.
    // If you don't see motion here, it's a motor/driver problem.

    // Optional hardware-interrupt panic button on DI-6.
    // Uncomment ONE of the lines below depending on which ClearCore
    // library API your installed version exposes:
    // ConnectorDI6.Mode(Connector::INPUT_DIGITAL);
    // ConnectorDI6.InterruptHandlerSet(&panicButtonISR, InputManager::FALLING, true);
}

void loop() {
    // Arduino Ethernet wrapper refreshes LwIP internally on each call
    serviceEstop();
    serviceModbus();         // the "panel stop" path
    serviceMotion();         // back-and-forth state machine
    updateStatusRegisters(); // refresh what the Pi will read next poll
}

// ── Bring-up ──────────────────────────────────────────────────────────────
void setupMotor() {
    MotorMgr.MotorInputClocking(MotorManager::CLOCK_RATE_NORMAL);
    MotorMgr.MotorModeSet(MotorManager::MOTOR_ALL,
                          Connector::CPM_MODE_STEP_AND_DIR);
    motor.VelMax(TEST_VEL_SPS);
    motor.AccelMax(TEST_ACCEL_SPSPS);
    motor.EnableRequest(true);
    delay(100);
    regs.state = CST_ENABLED;
    Serial.println("[MOTOR] M-0 enabled, step-and-dir");
    // Limit switch inputs
    homeLimit.Mode(Connector::INPUT_DIGITAL);
    farLimit.Mode(Connector::INPUT_DIGITAL);
    estopInput.Mode(Connector::INPUT_DIGITAL);
    Serial.println("[MOTOR] limit switches on IO-2 (home) and IO-3 (far) and e-stop on IO-1");
}

void setupEthernet() {
    Serial.print("[NET] PHY link");
    // Wait for the PHY to report a link. On some ClearCore firmware versions
    // linkStatus() may return Unknown (the W5200/W5500 detection mechanism
    // isn't applicable to ClearCore's integrated PHY) — treat Unknown as OK
    // and proceed. Only LinkOFF is a hard fail, and we time out anyway.
    uint32_t linkT0 = millis();
    while (Ethernet.linkStatus() == LinkOFF) {
        Serial.print(".");
        delay(500);
        if (millis() - linkT0 > 10000) {
            Serial.println(" TIMEOUT (continuing anyway)");
            break;
        }
    }
    if (Ethernet.linkStatus() != LinkOFF) {
        Serial.println(" up");
    }

    if (USE_DHCP) {
        if (Ethernet.begin(MAC_PLACEHOLDER) == 0) {
            Serial.println("[NET] DHCP FAIL - halting");
            while (true) continue;
        }
    } else {
        // Static: mac, ip, dns, gateway, subnet
        Ethernet.begin(MAC_PLACEHOLDER, STATIC_IP, DNS_IP, GATEWAY, NETMASK);
    }

    IPAddress ip = Ethernet.localIP();
    sprintf(dbg, "[NET] IP = %u.%u.%u.%u",
            (unsigned)ip[0], (unsigned)ip[1], (unsigned)ip[2], (unsigned)ip[3]);
    Serial.println(dbg);
}

// ── Modbus TCP server ─────────────────────────────────────────────────────
//
// Minimal single-client MBAP parser. The CLIENT_INFC struct is exposed as
// a flat uint16_t[] via reinterpret_cast. Writes past the command channel
// (word 6) are rejected with an ILLEGAL_ADDRESS exception so the Pi can't
// clobber the status channel.
void serviceModbus() {
    if (!modbusClient.connected()) {
        EthernetClient newClient = modbusServer.available();
        if (newClient) {
            modbusClient = newClient;
            Serial.println("[MODBUS] client connected");
            setMsg("PI CONNECTED");
        } else {
            return;
        }
    }

    int avail = modbusClient.available();
    if (avail < 8) return;    // need at least MBAP(7) + fc(1)

    uint8_t mbap[7];
    for (int i = 0; i < 7; i++) mbap[i] = modbusClient.read();

    uint16_t length = readU16(&mbap[4]);
    uint8_t  unitId = mbap[6];

    if (length < 2 || length > 253) {
        while (modbusClient.available() > 0) modbusClient.read();
        return;
    }

    uint8_t  pdu[253];
    uint16_t pduLen = length - 1;
    for (uint16_t i = 0; i < pduLen; i++) {
        // Wait for the rest of the PDU bytes to arrive. Guard against
        // a client that drops mid-frame so we don't spin forever.
        uint32_t waitT0 = millis();
        while (modbusClient.available() == 0) {
            if (!modbusClient.connected() || (millis() - waitT0) > 500) {
                return;
            }
        }
        pdu[i] = modbusClient.read();
    }

    if (unitId != MODBUS_UNIT_ID && unitId != 0xFF) return;

    uint8_t   fc      = pdu[0];
    uint16_t *rawRegs = reinterpret_cast<uint16_t*>(&regs);

    switch (fc) {
    case FC_READ_HOLDING: {
        if (pduLen < 5) { sendModbusException(mbap, fc, EX_ILLEGAL_ADDRESS); break; }
        uint16_t addr = readU16(&pdu[1]);
        uint16_t qty  = readU16(&pdu[3]);
        if (qty == 0 || qty > 125 || addr + qty > NUM_REGS) {
            sendModbusException(mbap, fc, EX_ILLEGAL_ADDRESS);
            break;
        }
        uint8_t resp[260];
        for (int i = 0; i < 6; i++) resp[i] = mbap[i];
        writeU16(&resp[4], 3 + 2 * qty);
        resp[6] = unitId;
        resp[7] = fc;
        resp[8] = (uint8_t)(2 * qty);
        for (uint16_t i = 0; i < qty; i++) {
            writeU16(&resp[9 + 2 * i], rawRegs[addr + i]);
        }
        modbusClient.write(resp, 9 + 2 * qty);
        break;
    }

    case FC_WRITE_SINGLE_REG: {
        if (pduLen < 5) { sendModbusException(mbap, fc, EX_ILLEGAL_ADDRESS); break; }
        uint16_t addr = readU16(&pdu[1]);
        uint16_t val  = readU16(&pdu[3]);
        // Only allow writes into the command channel (words 0..6)
        if (addr > W_ADDR(cmd)) {
            sendModbusException(mbap, fc, EX_ILLEGAL_ADDRESS);
            break;
        }
        rawRegs[addr] = val;
        if (addr == W_ADDR(cmd)) handleCommand(val);

        uint8_t resp[12];
        for (int i = 0; i < 7; i++) resp[i] = mbap[i];
        writeU16(&resp[4], 6);
        resp[7] = fc;
        writeU16(&resp[8], addr);
        writeU16(&resp[10], val);
        modbusClient.write(resp, 12);
        break;
    }

    case FC_WRITE_MULTIPLE_REG: {
        if (pduLen < 6) { sendModbusException(mbap, fc, EX_ILLEGAL_ADDRESS); break; }
        uint16_t addr = readU16(&pdu[1]);
        uint16_t qty  = readU16(&pdu[3]);
        uint8_t  bc   = pdu[5];
        if (qty == 0 || bc != 2 * qty) {
            sendModbusException(mbap, fc, EX_ILLEGAL_ADDRESS);
            break;
        }
        // Protect the status channel: writes must stay in words 0..6
        if (addr + qty > W_ADDR(cmd) + 1) {
            sendModbusException(mbap, fc, EX_ILLEGAL_ADDRESS);
            break;
        }
        for (uint16_t i = 0; i < qty; i++) {
            rawRegs[addr + i] = readU16(&pdu[6 + 2 * i]);
        }
        // Dispatch LAST so acc/vel/target are in place before handleCommand()
        if (addr <= W_ADDR(cmd) && W_ADDR(cmd) < addr + qty) {
            handleCommand(regs.cmd);
        }

        uint8_t resp[12];
        for (int i = 0; i < 7; i++) resp[i] = mbap[i];
        writeU16(&resp[4], 6);
        resp[7] = fc;
        writeU16(&resp[8], addr);
        writeU16(&resp[10], qty);
        modbusClient.write(resp, 12);
        break;
    }

    default:
        sendModbusException(mbap, fc, EX_ILLEGAL_FUNCTION);
        break;
    }
}

void sendModbusException(uint8_t *mbapHdr, uint8_t fc, uint8_t exCode) {
    uint8_t resp[9];
    for (int i = 0; i < 7; i++) resp[i] = mbapHdr[i];
    writeU16(&resp[4], 3);
    resp[7] = fc | 0x80;
    resp[8] = exCode;
    modbusClient.write(resp, 9);
}

// ── Command dispatch ──────────────────────────────────────────────────────
void handleCommand(uint16_t cmd) {
    sprintf(dbg, "[CMD] entry cmd=%u CCMD_RUN_1=%u", cmd, (unsigned)CCMD_RUN_1);
    Serial.println(dbg);

    if (estopEngaged() && cmd != CCMD_STOP && cmd != CCMD_DISAB_MTRS) {
        setMsg("CMD BLOCKED - E-STOP ENGAGED");
        regs.cmd = CCMD_NACK;
        return;
    }

    switch (cmd) {
    case CCMD_NONE:
        Serial.println("[CMD] -> NONE branch");
        return;

    case CCMD_NEXT_POINT:
        if (motionState != MS_AT_CAPTURE) {
            setMsg("NEXT_POINT ignored - not at capture");
            regs.cmd = CCMD_NACK;
            return;
        }
        // Advance one scan step from the current commanded position.
        // farLimitHit() is checked inside serviceMotion so the step aborts
        // cleanly if we hit the limit mid-move.
        capturePointIdx++;
        if (capturePointIdx >= SCAN_MAX_POINTS) {
            setMsg("SCAN POINT CAP - returning");
            motionState = MS_RETURNING;
            motor.Move(-MAX_TRAVEL_STEPS, StepGenerator::MOVE_TARGET_ABSOLUTE);
            break;
        }
        motor.Move(motor.PositionRefCommanded() + SCAN_STEP_STEPS,
                   StepGenerator::MOVE_TARGET_ABSOLUTE);
        motionState = MS_SCAN_STEP;
        sprintf(dbg, "SCAN PT %u -> +%ld", capturePointIdx, (long)SCAN_STEP_STEPS);
        setMsg(dbg);
        break;
        
    case CCMD_ENAB_MTRS:
        Serial.println("[CMD] -> ENAB branch");
        motor.EnableRequest(true);
        regs.state = CST_ENABLED;
        setMsg("MOTORS ENABLED");
        break;

    case CCMD_DISAB_MTRS:
        motor.MoveStopAbrupt();
        motor.EnableRequest(false);
        motionState = MS_DISABLED;
        regs.state  = CST_IDLE;
        setMsg("MOTORS DISABLED");
        break;

    case CCMD_SET_ZERO:
        // Future: motor.PositionRefSet(0)
        setMsg("SET_ZERO (nop in test)");
        break;

    case CCMD_MOVE: {
        // Use acc/vel/target_posn written in the same transaction
        uint32_t velMag = (regs.vel < 0) ? (uint32_t)(-regs.vel) : (uint32_t)regs.vel;
        motor.VelMax(velMag ? velMag : (uint32_t)TEST_VEL_SPS);
        motor.AccelMax(regs.acc ? regs.acc : (uint32_t)TEST_ACCEL_SPSPS);
        motor.EnableRequest(true);
        motor.Move(regs.target_posn, StepGenerator::MOVE_TARGET_ABSOLUTE);
        motionState = MS_STOPPED;   // MOVE is a one-shot, not the sequence
        regs.state  = CST_RUNNING;
        sprintf(dbg, "MOVE to %ld", (long)regs.target_posn);
        setMsg(dbg);
        break;
    }

    case CCMD_STOP:
        motor.MoveStopAbrupt();
        motionState = MS_STOPPED;
        regs.state  = CST_STOPPED;
        setMsg("STOP RX");
        Serial.println("[MOTION] STOP");
        break;

    case CCMD_RUN_1:
        Serial.println("[CMD] -> RUN_1 branch, calling startSequence1()");
        startSequence1();
        Serial.println("[CMD] -> returned from startSequence1()");
        break;

    default:
        sprintf(dbg, "[CMD] -> DEFAULT branch, unknown %u", cmd);
        Serial.println(dbg);
        regs.cmd = CCMD_NACK;
        return;
    }

    Serial.println("[CMD] exit, clearing latch");
    regs.cmd = CCMD_NONE;
}

// ── Sequence 1: home → scan → return ──────────────────────────────────────
void startSequence1() {
    Serial.println("[SEQ1] >>> startSequence1() called");
    sprintf(dbg, "[SEQ1] limit reads: home=%d far=%d  motionState=%d",
            homeLimitHit(), farLimitHit(), (int)motionState);
    Serial.println(dbg);

    // Clean enable cycle
    motor.EnableRequest(false);
    delay(50);   // brief; OK because <500ms Modbus timeout
    motor.EnableRequest(true);

    motor.VelMax(SCAN_VEL_SPS);
    motor.AccelMax(SCAN_ACCEL_SPSPS);
    motor.PositionRefSet(0);

    // Start the rearm wait; serviceMotion() will transition to homing
    // once the drive has settled.
    motionState = MS_REARM_WAIT;
    rearmStartMs = millis();
    regs.state = CST_RUNNING;
    setMsg("SEQ1: rearming drive");
}

// ── Motion state machine ──────────────────────────────────────────────────
void serviceMotion() {
    // Panic button ISR → flag → here (handled outside interrupt context)
    if (stopRequested) {
        stopRequested = false;
        motor.MoveStopAbrupt();
        motor.EnableRequest(false);
        motionState = MS_DISABLED;
        regs.state  = CST_FAULT;
        setMsg("PANIC BUTTON");
        Serial.println("[ISR] panic button handled");
    }

    switch (motionState) {

    case MS_REARM_WAIT:
        if (millis() - rearmStartMs >= 2000) {
            Serial.println("[SEQ1] rearm complete, starting motion");
            capturePointIdx = 0;
            if (homeLimitHit()) {
                motionState = MS_SCAN_STEP;
                motor.Move(SCAN_STEP_STEPS, StepGenerator::MOVE_TARGET_ABSOLUTE);
                setMsg("SEQ1: already home, first step");
            } else {
                motionState = MS_HOMING;
                motor.Move(-MAX_TRAVEL_STEPS, StepGenerator::MOVE_TARGET_ABSOLUTE);
                setMsg("SEQ1: homing");
            }
        }
        break;


    case MS_HOMING:
        // Drive toward home switch; stop the instant it trips.
        if (homeLimitHit()) {
            motor.MoveStopAbrupt();
            motor.PositionRefSet(0);
            capturePointIdx = 0;
            setMsg("HOMED - starting scan");
            motionState = MS_SCAN_STEP;
            motor.Move(SCAN_STEP_STEPS, StepGenerator::MOVE_TARGET_ABSOLUTE);
        }
        } else if (motor.StepsComplete()) {
            // Ran out of travel without hitting the switch — bail out.
            setMsg("HOME FAIL - no limit hit");
            Serial.println("[SEQ1] home FAIL");
            motionState = MS_STOPPED;
            regs.state  = CST_FAULT;
        }
        break;

    case MS_SCAN_STEP:
        // Advancing one step toward far limit. Two ways to exit:
        if (farLimitHit()) {
            motor.MoveStopAbrupt();
            setMsg("FAR LIMIT - returning");
            Serial.println("[SEQ1] far limit hit -> returning");
            motionState = MS_RETURNING;
            motor.Move(-MAX_TRAVEL_STEPS, StepGenerator::MOVE_TARGET_ABSOLUTE);
        } else if (motor.StepsComplete()) {
            // Reached target; begin settle timer
            motionState = MS_SETTLE;
            settleStartMs = millis();
        }
        break;

    case MS_SETTLE:
        // Hold still long enough for mechanical ring-down to die off.
        // Abort settle if far limit went high during ring-down (edge case).
        if (farLimitHit()) {
            motionState = MS_RETURNING;
            motor.Move(-MAX_TRAVEL_STEPS, StepGenerator::MOVE_TARGET_ABSOLUTE);
            setMsg("FAR LIMIT during settle - returning");
            break;
        }
        if (millis() - settleStartMs >= SCAN_SETTLE_MS) {
            motionState = MS_AT_CAPTURE;
            sprintf(dbg, "AT CAPTURE pt=%u pos=%ld",
                    capturePointIdx, (long)motor.PositionRefCommanded());
            setMsg(dbg);
        }
        break;

case MS_AT_CAPTURE:
    // Nothing to do — Pi will poll status, see at_capture=1, grab
    // averaged frames, then send CCMD_NEXT_POINT to advance. The command
    // handler transitions us back to MS_SCAN_STEP.
    break;

    case MS_RETURNING:
        // Slow return back to home, then park.
        if (homeLimitHit()) {
            motor.MoveStopAbrupt();
            motor.PositionRefSet(0);
            setMsg("SCAN COMPLETE - parked at home");
            Serial.println("[SEQ1] complete, parked at home");
            motionState = MS_STOPPED;
            regs.state  = CST_STOPPED;
        } else if (motor.StepsComplete()) {
            setMsg("RETURN FAIL - no home limit");
            Serial.println("[SEQ1] return FAIL");
            motionState = MS_STOPPED;
            regs.state  = CST_FAULT;
        }
        break;

    default:
        break;   // MS_IDLE / MS_STOPPED / MS_DISABLED — nothing to service
    }
}

// ── Status channel refresh ────────────────────────────────────────────────
void updateStatusRegisters() {
    CCMTR_STATUS s;
    s.b.drvs_enabled = (motionState != MS_DISABLED) ? 1 : 0;
    s.b.moving       = !motor.StepsComplete() ? 1 : 0;
    s.b.scanning     = (motionState == MS_REARM_WAIT ||
                        motionState == MS_HOMING     ||
                        motionState == MS_SCAN_STEP  ||
                        motionState == MS_SETTLE     ||
                        motionState == MS_AT_CAPTURE ||
                        motionState == MS_RETURNING) ? 1 : 0;
    s.b.at_capture   = (motionState == MS_AT_CAPTURE) ? 1 : 0;
    s.b.fault        = (regs.state == CST_FAULT) ? 1 : 0;
    s.b.estop        = estopEngaged() ? 1 : 0;
    s.b.homed        = 0;
    s.b.hlfb         = 0;
    regs.status = s;

    regs.cur_posn = motor.PositionRefCommanded();
}

// ── msg buffer helper ─────────────────────────────────────────────────────
// The Pi polls msg_cnt as part of the normal status read; when it sees
// the counter tick, it does a separate burst-read of the 64-byte msg
// buffer and routes the contents into its log panel.
void setMsg(const char *text) {
    memset(regs.msg, 0, sizeof(regs.msg));
    strncpy(regs.msg, text, sizeof(regs.msg) - 1);
    regs.msg_cnt++;
}

// ── Panic button ISR ──────────────────────────────────────────────────────
// Runs in interrupt context. Keep it TINY — no Ethernet, no USB, no printf.
// Just set a flag; serviceMotion() does the real work next loop iteration.
void panicButtonISR() {
    stopRequested = true;
}

void serviceEstop() {
    static bool prevEngaged = false;
    bool engaged = estopEngaged();

    if (engaged && !prevEngaged) {
        // Rising edge — just got pressed
        motor.MoveStopAbrupt();
        motor.EnableRequest(false);
        motionState = MS_DISABLED;
        regs.state  = CST_FAULT;
        setMsg("E-STOP ENGAGED");
        Serial.println("[ESTOP] engaged");
    } else if (!engaged && prevEngaged) {
        // Released — don't auto-enable, operator must re-arm from UI
        motor.ClearAlerts();       // clear any drive-side latched alert
        setMsg("E-STOP CLEARED - press ENABLE to rearm");
        Serial.println("[ESTOP] released");
        regs.state = CST_IDLE;
        motionState = MS_STOPPED;   // <-- add this; clears MS_DISABLED

    }
    prevEngaged = engaged;
}
