pub mod a1;
#[cfg(feature = "lie-generated")]
pub mod bcd;
pub mod counting_alloc;
pub mod lie;

/// Only global-state measurements/lifecycle checks serialize; value tests stay parallel.
#[allow(dead_code)]
pub fn serial() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    SERIAL.lock().unwrap_or_else(|poison| poison.into_inner())
}
