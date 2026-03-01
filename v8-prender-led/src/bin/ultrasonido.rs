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

        let pulse_us = (end - start).max(0) as f32;
        let dist_cm = pulse_us / 58.0;

        println!("pulse_us={:.0} dist_cm={:.1}", pulse_us, dist_cm);

        FreeRtos::delay_ms(60);
    }
}
