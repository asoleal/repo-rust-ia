# rust-ia-1 (ESP32 Xtensa + Rust + ESP-IDF)
Este proyecto contiene:
- `ia_test`: inferencia con modelo ya entrenado (modelo en `assets/model.bin`).
- `ia_train`: inferencia + **entrenamiento manual** (solo capa Dense) y guardado en **NVS** para que el ajuste permanezca después de reiniciar.

Está pensado para ESP32 clásico (Xtensa) y se compila con el toolchain de `espup`.

---

## 0) Requisitos y contexto
### Hardware
- ESP32 (con USB-UART CP2102 en tu caso).
- Sensor ultrasónico con:
  - TRIG: GPIO23
  - ECHO: GPIO2

### Software (PC)
- Rust + cargo
- Toolchain Xtensa/ESP-IDF vía `espup` (que crea un script `export-esp.sh`)
- `espflash` para flashear y abrir monitor serial

---

## 1) Activar entorno Xtensa (OBLIGATORIO en cada terminal)
Primero ubicamos el script:

```bash
find ~ -maxdepth 3 -name 'export-esp.sh' -type f 2>/dev/null

En tu máquina quedó en:

bash
/home/jjlg/export-esp.sh

Actívalo en la terminal actual:

bash
. /home/jjlg/export-esp.sh

Verifica que el compilador Xtensa quedó disponible:

bash
which xtensa-esp32-elf-gcc

Debe mostrar algo parecido a:

text
/home/jjlg/.rustup/toolchains/esp/xtensa-esp-elf/.../bin/xtensa-esp32-elf-gcc

2) Compilar el proyecto (sanear build si hay problemas)

Dentro de este repo:

bash
cd /home/jjlg/repo-rust-ia/rust-ia-1

Si alguna vez te falla por residuos, limpia y recompila:

bash
rm -rf target .embuild
cargo build --bin ia_test

3) Puerto serial estable (recomendado)

Usamos el symlink estable:

bash
ls -l /dev/serial/by-id 2>/dev/null

En tu caso (ejemplo):

bash
PORT="/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"

4) Flashear y monitor

Compila el binario y flashea:

bash
cargo build --bin ia_train
espflash flash --port "$PORT" --monitor target/xtensa-esp32-espidf/debug/ia_train

Notas del monitor:

    Ctrl+C sale del monitor.

    Ctrl+R normalmente resetea el chip (si tu configuración lo soporta).

5) Modelo y arquitectura (lo que usa el firmware)

El firmware carga el modelo base desde:

    assets/model.bin (embebido con include_bytes!)

Ese binario representa una red simple:

    Entrada: secuencia de longitud 128 (1 canal)

    Conv1D:

        filtros: 12

        kernel: 5

        stride: 1

    Activación: LeakyReLU

    Average pooling 1D: size=2

    Dense final: 744 -> 3 (3 clases)

Las clases en el firmware están mapeadas así:

    clase 0: quieto

    clase 1: acercando

    clase 2: alejando

6) ia_train: qué hace exactamente

ia_train hace:

    Captura una ventana de 128 muestras del ultrasónico.

    Ejecuta forward (conv + pooling + dense).

    Imprime:

        distancia, logits out=[...], clase estimada y si está en modo entrenamiento.

    Si activas modo entrenamiento manual:

        con 1/2/3 aplicas 1 update SGD SOLO a Dense

        guarda Dense actualizado en NVS (blob) con clave dense_v1

Importante:

    No se “re-entrena” el modelo completo; solo ajusta Dense (más realista en dispositivo).

    Lo guardado en NVS debe recargarse al reiniciar.

7) Entrenamiento MANUAL (flujo correcto)
7.1 Iniciar programa

Flashea y abre monitor:

bash
cargo build --bin ia_train
espflash flash --port "$PORT" --monitor target/xtensa-esp32-espidf/debug/ia_train

Al inicio verás algo como:

    loaded dense from NVS (...) si ya entrenaste antes

    o no dense in NVS yet si es primera vez

7.2 Modo normal (solo inferencia)

Por defecto:

    training_mode=false

    no entrena aunque presiones números

7.3 Activar entrenamiento

En el monitor presiona:

    t

Debe imprimir:

    training_mode=true

7.4 Etiquetar con 1 / 2 / 3 (esto entrena + guarda)

En modo entrenamiento, presiona SOLO una tecla:

    1 = etiqueta clase 0 = quieto

    2 = etiqueta clase 1 = acercando

    3 = etiqueta clase 2 = alejando

Cada tecla aplica:

    softmax(out)

    grad = (p - y)

    SGD sobre W y b de Dense

    nvs.set_blob("dense_v1", ...)

Debería imprimir:

    trained+saved (... bytes) label=1/2/3

7.5 Desactivar entrenamiento

Presiona:

    t

y vuelves a:

    training_mode=false

7.6 Verificar persistencia (NVS)

    Reinicia (o vuelve a correr el monitor).

    Debe decir loaded dense from NVS ....

8) Ajuste de latencia (por qué decía “quieto” cuando te movías despacio)

La latencia depende de:

    seq_len = 128 muestras por ventana

    el delay entre muestras dentro de:

rust
for i in 0..seq_len {
    ...
    FreeRtos::delay_ms(X);
}

Si X=60, una ventana tarda aprox 7.7s (muy lento).
Lo bajamos a:

    FreeRtos::delay_ms(20);

Eso baja la ventana a ~2.6s y responde mucho mejor.

OJO:

    El delay del bloque de ERROR del modelo (cuando flat.len() != total_need) NO afecta la latencia normal; solo corre si el modelo está mal.

9) Archivos importantes del proyecto

    src/bin/ia_train.rs -> inferencia + entrenamiento manual + NVS

    assets/model.bin -> pesos base (conv + dense)

    src/main.rs o src/bin/ia_test.rs (según tu repo) -> inferencia base (si existe)

10) Comandos útiles (resumen operativo)

Activar toolchain:

bash
. /home/jjlg/export-esp.sh
which xtensa-esp32-elf-gcc

Compilar:

bash
cd /home/jjlg/repo-rust-ia/rust-ia-1
cargo build --bin ia_train

Flashear + monitor:

bash
PORT="/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"
espflash flash --port "$PORT" --monitor target/xtensa-esp32-espidf/debug/ia_train

Entrenar en monitor:

    t para activar/desactivar training

    1 quieto, 2 acercando, 3 alejando

11) Próximos pasos (si quieres mejorar más)

    Ventana deslizante (ring buffer) para inferencia cada 16/32 muestras (latencia aún menor sin reducir tanto el delay).

    Guardar con commit explícito si hiciera falta (depende de cómo esté configurado el wrapper/NVS).

    Mejorar entrada: usar diferencia (velocidad) además de distancia para detectar “acercando/alejando” más robusto.
