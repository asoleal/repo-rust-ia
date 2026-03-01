# Rust + ESP32 (clásico) en Arch Linux — LED + Ultrasonido

Este repo documenta cómo configuramos un entorno en Arch Linux para programar un **ESP32 clásico (Xtensa)** con Rust usando **ESP-IDF** (vía `esp-idf-template`), cómo verificamos conexión/puerto serial, y cómo corrimos dos programas:
- Blink de LED en GPIO27.
- Lectura de ultrasonido (TRIG=GPIO23, ECHO=GPIO2) como binario separado.

---

## Requisitos (PC)

- Arch Linux.
- Rust con `rustup` (toolchain estable).
- `espup` (toolchain Xtensa para ESP32).
- `ldproxy` (linker wrapper requerido por el template).
- `espflash` (para flashear y monitor serial).

Instalamos:
```bash
sudo pacman -S --needed rustup
rustup default stable

# espup (toolchain ESP32)
cargo install espup --locked

# linker wrapper
cargo install ldproxy

# espflash (en Arch lo instalamos con pacman para evitar errores de compilación)
sudo pacman -S --needed espflash

# para generar el proyecto
# (ya lo teníamos instalado, pero este es el comando)
cargo install cargo-generate
```

---

## Activar el entorno de ESP (Xtensa)

Verificamos que existía el script creado por `espup`:

```bash
ls -l ~/export-esp.sh
```

Luego, en cada terminal donde vayas a compilar/flashear:

```bash
. ~/export-esp.sh
```

Comprobación rápida de toolchain Xtensa:

```bash
which xtensa-esp32-elf-gcc
```

---

## Crear el proyecto (esp-idf-template)

Creamos el proyecto con `cargo-generate`:

```bash
cargo generate esp-rs/esp-idf-template cargo
```

- Nombre del proyecto: `v8-prender-led`
- MCU: `esp32`
- Opciones avanzadas: `false`

Entrar al proyecto:

```bash
cd ~/repo-rust-ia/v8-prender-led
```

---

## Verificar conexión del ESP32 por USB/serial

### 1) Confirmar que el USB se ve
Con la placa conectada:

```bash
lsusb
```

En nuestro caso apareció como:
- `Silicon Labs CP210x UART Bridge` (CP2102).

### 2) Encontrar el puerto serial
Buscamos el puerto tty:

```bash
ls -l /dev/serial/by-id/ 2>/dev/null
ls -l /dev/ttyUSB* 2>/dev/null
```

Nos dio `/dev/ttyUSB1` y un symlink estable en `/dev/serial/by-id/...`.

### 3) Permisos (grupo uucp)
El dispositivo estaba con grupo `uucp` (modo típico `crw-rw----`), así que agregamos el usuario:

```bash
sudo usermod -aG uucp $USER
groups
```

(En algunos sistemas toca cerrar sesión y volver a entrar para que aplique.)

### 4) Probar que espflash se conecta
Con el puerto conocido:

```bash
espflash board-info --port /dev/ttyUSB1
```

Esto confirmó chip `esp32`, cristal 40 MHz, etc.

---

## Compilar el template (y el error de ldproxy)

Al compilar por primera vez apareció:

- `error: linker 'ldproxy' not found`

Solución:
```bash
cargo install ldproxy
```

Nota: ejecutar `ldproxy` directamente puede “panic” porque está pensado para ser invocado por Cargo como linker; lo importante es que `cargo build` lo encuentre en el PATH.

---

## Programa 1: LED en GPIO27 (binario principal)

Este código vive en `src/main.rs` y hace blink en GPIO27.

### Crear/editar el archivo

```bash
cd ~/repo-rust-ia/v8-prender-led

cat > src/main.rs <<'EOF2'
use anyhow::Result;
use esp_idf_svc::{
    hal::{delay::FreeRtos, gpio::PinDriver, peripherals::Peripherals},
    sys::link_patches,
};

fn main() -> Result<()> {
    link_patches();

    let peripherals = Peripherals::take().unwrap();
    let mut led = PinDriver::output(peripherals.pins.gpio27)?;

    loop {
        led.set_high()?;
        FreeRtos::delay_ms(500);
        led.set_low()?;
        FreeRtos::delay_ms(500);
    }
}
EOF2
```

### Dependencia `anyhow`
Si no está, se agrega así:

```bash
cargo add anyhow
```

### Compilar y flashear
Compilar:

```bash
cargo build --bin v8-prender-led
```

Flashear y abrir monitor:

```bash
espflash flash --port /dev/ttyUSB1 --monitor target/xtensa-esp32-espidf/debug/v8-prender-led
```

---

## Programa 2: Ultrasonido (TRIG=23, ECHO=2) sin borrar el blink

Para no sobrescribir `src/main.rs`, creamos un segundo binario en `src/bin/ultrasonido.rs`.
Cargo trata cada archivo en `src/bin/*.rs` como un ejecutable distinto.

### Crear el archivo

```bash
cd ~/repo-rust-ia/v8-prender-led
mkdir -p src/bin

cat > src/bin/ultrasonido.rs <<'EOF3'
use anyhow::Result;
use esp_idf_svc::{
    hal::{delay::{Ets, FreeRtos}, gpio::PinDriver, peripherals::Peripherals},
    sys::{esp_timer_get_time, link_patches},
};

fn micros() -> i64 {
    unsafe { esp_timer_get_time() }
}

fn main() -> Result<()> {
    link_patches();

    let peripherals = Peripherals::take().unwrap();
    let mut trig = PinDriver::output(peripherals.pins.gpio23)?;
    let echo = PinDriver::input(peripherals.pins.gpio2)?;

    trig.set_low()?;
    FreeRtos::delay_ms(50);

    loop {
        // Pulso TRIG: HIGH 10us
        trig.set_low()?;
        Ets::delay_us(2);
        trig.set_high()?;
        Ets::delay_us(10);
        trig.set_low()?;

        // Esperar ECHO HIGH (timeout ~30ms)
        let t0 = micros();
        while echo.is_low() && (micros() - t0) <= 30_000 {}

        // Medir duración ECHO HIGH (timeout ~30ms)
        let start = micros();
        while echo.is_high() && (micros() - start) <= 30_000 {}
        let end = micros();

        // HC-SR04 típico: cm ≈ us/58
        let pulse_us = (end - start).max(0) as f32;
        let dist_cm = pulse_us / 58.0;

        println!("pulse_us={:.0} dist_cm={:.1}", pulse_us, dist_cm);

        FreeRtos::delay_ms(60);
    }
}
EOF3
```

### Nota sobre `delay_us`
Nos apareció que `FreeRtos::delay_us()` era privado; lo resolvimos usando:
- `Ets::delay_us(...)` para microsegundos (pulso TRIG).
- `FreeRtos::delay_ms(...)` para esperas largas.

### Compilar y flashear este binario
Compilar:

```bash
cargo build --bin ultrasonido
```

Flashear:

```bash
espflash flash --port /dev/ttyUSB1 --monitor target/xtensa-esp32-espidf/debug/ultrasonido
```

---

## Ver código rápido

```bash
cat src/main.rs
cat src/bin/ultrasonido.rs
```

---

## Tips / notas útiles

- Para evitar problemas de compilación de herramientas, en Arch instalamos `espflash` con `pacman`.
- Usa `/dev/serial/by-id/...` si quieres un puerto estable (cuando cambie el número ttyUSB).
- Si el sensor ultrasónico es tipo HC-SR04 a 5V: ojo con ECHO a 5V (ideal usar divisor/level shifting hacia 3.3V). Si ya lo probaste y funciona, igual es buena práctica proteger el pin.

---
