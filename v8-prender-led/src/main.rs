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
