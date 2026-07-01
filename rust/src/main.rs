use std::net::UdpSocket;
use ndarray::Array2;
use ort::{inputs, session::Session};


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/Users/shivaram/telemetry-anomaly-detector/Model/model/telemetry_autoencoder.onnx";

    let mut session = Session::builder()?
    .commit_from_file(model_path)?;

    let socket = UdpSocket::bind("127.0.0.1:0")?;

    let message = "Hey from Rust";

    let server_address  = "127.0.0.1:8008";

    socket.send_to(message.as_bytes(), server_address).expect("Could not send the message");

    println!("Message sent to python initiating the sending process!");

    let mut buf = [0;60];
    for _ in 0..1{
        let (amt, _src) = socket.recv_from(& mut buf)?;
        if amt == 60 {
            let composite_id = u32::from_le_bytes(buf[0..4].try_into().unwrap());
            let float: Vec<f32> = buf[4..60]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
            
            let array = Array2::<f32>::from_shape_vec((1, 14), float).unwrap();
            
            let outputs = session.run(inputs![array.clone()])?;
        }
    }

    Ok(())
}
