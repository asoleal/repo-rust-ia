# Repo Rust + ESP32 + IA (ultrasonido)

Este repo documenta el proceso completo que seguimos para:

1) Preparar un proyecto Rust para ESP32 (ESP-IDF + Rust).
2) Leer un sensor ultrasónico (HC-SR04 / compatible) y ver datos por puerto serial.
3) Crear y auditar una mini-red neuronal hecha a mano en Rust (Conv1D + activación + Pool + Dense).
4) Entrenar en PC, exportar el modelo a `model.bin` y ejecutar inferencia en el ESP32 usando `include_bytes!`.
5) Preparar el camino para alertas y (más adelante) persistencia con NVS.

> Nota: este repo evita subir caches pesados de build (`target/`, `.embuild/`, `.espressif/`, `build/`, etc.) por historial y tamaño.

---

## 0) Entorno

- OS: Linux
- Placa: ESP32 clásico (Xtensa) detectado con `espflash board-info`
- USB-Serial: Silicon Labs CP2102 (aparece como `/dev/ttyUSB0` o symlink en `/dev/serial/by-id/`)
- Stack: ESP-IDF + Rust (template `esp-idf-template`)
- Flasheo/monitor: `espflash`

---

## 1) Clonado limpio y verificación de Git

(Esto se hizo para limpiar historial/estado por carpetas pesadas subidas por error).

Comandos útiles para comparar local vs remoto sin borrar nada:

```bash
git fetch --prune origin
git log -1 --oneline --decorate main
git log -1 --oneline --decorate origin/main
git diff --name-status origin/main..main
git diff origin/main..main
git diff --name-status origin/main..main -- v8-prender-led/
git diff origin/main..main -- v8-prender-led/
git ls-tree -r --name-only main -- v8-prender-led/
git ls-tree -r --name-only origin/main -- v8-prender-led/
git status --porcelain=v1 --untracked-files=all -- v8-prender-led/
git status --ignored --porcelain=v1 -- v8-prender-led/
git check-ignore -v v8-prender-led/** | head -n 50

2) Preparar toolchain ESP32 (ESP-IDF + Rust)

Instalación (según tu distro; ejemplo Arch):

bash
sudo pacman -S --needed rustup
rustup default stable

cargo install espup --locked
cargo install ldproxy
sudo pacman -S --needed espflash
cargo install cargo-generate

Activar entorno de ESP (en cada terminal donde compiles/flashees):

bash
ls -l export-esp.sh
. ./export-esp.sh
which xtensa-esp32-elf-gcc

3) Conexión serial y detección de chip

Encontrar puerto:

bash
ls -l /dev/serial/by-id 2>/dev/null || true
ls -l /dev/ttyUSB* 2>/dev/null || true
ls -l /dev/ttyACM* 2>/dev/null || true

Configurar variable:

bash
PORT="/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"
ls -l "$PORT"

Permisos (Arch suele usar uucp):

bash
ls -l /dev/ttyUSB0
groups
sudo usermod -aG uucp "$USER"
# luego cerrar sesión y volver a entrar

Detectar chip:

bash
espflash board-info --port "$PORT"

Salida esperada (ejemplo real):

    Chip type: esp32 (revision v1.1)

    Flash size: 16MB

4) Proyecto ESP32: “tick” por serial (sanity check)

En el firmware (proyecto ESP32), pusimos un loop para ver salida continua por monitor:

rust
use std::thread;
use std::time::Duration;
use esp_idf_svc::sys::link_patches;

fn main() {
    link_patches();

    let mut n: u32 = 0;
    loop {
        println!("tick {}", n);
        n = n.wrapping_add(1);
        thread::sleep(Duration::from_millis(500));
    }
}

Compilar y flashear:

bash
cargo build
espflash flash --port "$PORT" --monitor target/xtensa-esp32-espidf/debug/<binario>

5) Ultrasonido (HC-SR04): lectura básica

Cableado típico (ejemplo):

    TRIG -> GPIO23

    ECHO -> GPIO2

⚠️ Importante: si tu HC-SR04 está a 5V, el pin ECHO puede salir a 5V. Se recomienda divisor/level shifter para 3.3V.

Firmware: genera pulso TRIG y mide ECHO con timeout, calcula:

    pulse_us = duración del pulso ECHO

    dist_cm = pulse_us / 58.0

6) IA en Rust (PC): auditoría y cambios clave
6.1 Conv1D (validación)

Auditamos Conv1D:

    Forward y backward son consistentes para conv “valid” con stride 1.

    Observación importante: el input está tratado como shape (in_channels, len).

6.2 Pooling: cambio a AveragePool

Inicialmente había un MaxPool1D.forward pero sin backward correcto.
Para embedded y entrenamiento simple, decidimos usar AvgPool1D:

    Forward: promedio por ventana.

    Backward: distribuye gradiente dividiendo por size.
    Esto evita máscara/argmax (menos RAM/estado).

7) Entrenamiento en PC y exportación a model.bin

Creamos un proyecto de PC (ejemplo: v9-rust-ia/) donde:

    Implementamos Conv1D + LeakyReLU + AvgPool1D + Dense

    Entrenamos con dataset sintético tipo “ultrasonido” (tendencia + ruido + outliers)

    Cambiamos a 3 clases: quieto, acercando, alejando

    Guardamos el modelo como binario de f32 little-endian: model.bin

El formato final del modelo (en float32) es un vector plano:

    Conv weights: filters * in_ch * k

    Conv bias: filters

    Dense weights: dense_in * dense_out

    Dense bias: dense_out

Con parámetros:

    seq_len=128

    filters=12

    kernel=5

    conv_out_len=124

    pool_len=62

    dense_in=744

    dense_out=3

Tamaño esperado:

    conv_w = 60

    conv_b = 12

    dense_w = 2232

    dense_b = 3
    Total = 2307 f32 = 9228 bytes

En PC, verificamos que el model.bin se genera:

bash
cargo run --release
ls -l model.bin

8) Ejecutar IA en ESP32 (Opción A: include_bytes!)
8.1 Copiar el modelo al firmware

Copiar model.bin entrenado en PC al proyecto ESP32:

bash
mkdir -p rust-ia/assets
cp v9-rust-ia/model.bin rust-ia/assets/model.bin
ls -l rust-ia/assets/model.bin

8.2 Binario separado para no chocar con otros

Para evitar romper main.rs o el binario del ultrasonido, creamos un binario extra:

    rust-ia/src/bin/ia_test.rs

Este binario:

    Carga assets/model.bin con include_bytes!

    Convierte bytes a Vec<f32>

    Valida tamaño (flat.len == 2307)

    Lee una ventana real del ultrasonido (128 muestras, delay 60ms)

    Normaliza 0..200cm -> 0..1

    Ejecuta el forward (Conv + Leaky + AvgPool + Dense)

    Imprime:

        dist_cm=...

        out=[a,b,c]

        class=... (quieto/acercando/alejando)

Compilar y flashear ese binario:

bash
cargo build --bin ia_test
espflash flash --port "$PORT" --monitor target/xtensa-esp32-espidf/debug/ia_test

9) Cómo saber que está “usando la IA”

Se confirma por:

    model.bin bytes=... y flat.len=... need=... (carga correcta).

    out=[...] y class=... cambian al mover el objeto.

    Prueba A/B: cambiar model.bin y ver cambio de salida.

10) Alertas después de detección (viable)

Una vez tenemos class, podemos disparar alertas:

    Serial (logs)

    GPIO (LED/buzzer)

    Wi-Fi (HTTP/MQTT) si se configura red

Recomendaciones:

    Debounce (N ventanas consecutivas antes de alertar)

    Rate-limit (1 alerta cada X segundos)

    Umbral mínimo de confianza (diferencia entre scores)

11) Gitignore recomendado

Ignorar caches pesados y permitir subir el modelo binario del firmware:

text
/target/
**/target/
**/.embuild/
**/.espressif/
**/build/
**/sdkconfig
**/sdkconfig.old

# subir el modelo:
!rust-ia/assets/model.bin

12) Estado actual

    ESP32 lee ultrasonido, llena ventanas de 128 samples.

    IA corre en el ESP32 con modelo preentrenado en PC, cargado por include_bytes!.

    Siguiente paso: mejorar dataset real (capturar mediciones reales por serial), ajustar entrenamiento y luego (opcional) persistir modelo en NVS (Opción B).


