use std::sync::{atomic::{AtomicBool, Ordering}, Arc};
use std::time::Duration;
use tokio::io::AsyncWriteExt as _;
use tokio::net::TcpStream;
use tokio::sync::mpsc;

/// Encode an OPC "Set Pixel Colors" packet.
/// Protocol: [channel(1)][command=0(1)][len_hi(1)][len_lo(1)][R,G,B × n_leds]
fn build_packet(channel: u8, rgb: &[u8]) -> Vec<u8> {
    let len = rgb.len() as u16;
    let mut pkt = Vec::with_capacity(4 + rgb.len());
    pkt.push(channel);
    pkt.push(0x00); // command: set pixel colors
    pkt.push((len >> 8) as u8);
    pkt.push(len as u8);
    pkt.extend_from_slice(rgb);
    pkt
}

/// Spawn one async task per FadeCandy device.
/// Returns an `mpsc::Sender<Vec<u8>>` whose payload is the raw RGB bytes for that device's LEDs.
/// The channel is bounded to 2 so the renderer can drop frames on backpressure.
pub fn spawn(
    address: String,
    running: Arc<AtomicBool>,
) -> mpsc::Sender<Vec<u8>> {
    let (tx, rx) = mpsc::channel::<Vec<u8>>(2);
    tokio::spawn(opc_task(address, rx, running));
    tx
}

async fn opc_task(address: String, mut rx: mpsc::Receiver<Vec<u8>>, running: Arc<AtomicBool>) {
    let mut backoff = Duration::from_secs(2);

    while running.load(Ordering::Relaxed) {
        // --- connect with exponential backoff ---
        let addr_with_port = if address.contains(':') {
            address.clone()
        } else {
            format!("{address}:7890")
        };

        let mut stream = loop {
            if !running.load(Ordering::Relaxed) {
                return;
            }
            match TcpStream::connect(&addr_with_port).await {
                Ok(s) => {
                    s.set_nodelay(true).ok();
                    tracing::info!("OPC connected to {addr_with_port}");
                    backoff = Duration::from_secs(2); // reset on success
                    break s;
                }
                Err(e) => {
                    tracing::warn!(
                        "OPC connect to {addr_with_port} failed: {e} — retrying in {backoff:?}"
                    );
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(30));
                }
            }
        };

        // --- send loop ---
        while let Some(rgb) = rx.recv().await {
            if !running.load(Ordering::Relaxed) {
                return;
            }
            let packet = build_packet(0, &rgb);
            if let Err(e) = stream.write_all(&packet).await {
                tracing::warn!("OPC write to {addr_with_port} failed: {e} — reconnecting");
                break; // reconnect
            }
        }
    }
}
