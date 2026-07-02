use std::net::UdpSocket;
use ndarray::Array2;
use ort::{inputs, session::Session};
use ort::value::TensorRef;
// use std::time::Instant;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let start = Instant::now();
    let model_path = "/Users/shivaram/telemetry-anomaly-detector/Model/model/telemetry_autoencoder.onnx";
    let features_count = 14_f32;

    let mut session = Session::builder()?
    .commit_from_file(model_path)?;

    let socket = UdpSocket::bind("127.0.0.1:0")?;

    let message = "Hey from Rust";

    let server_address  = "127.0.0.1:8008";

    socket.send_to(message.as_bytes(), server_address).expect("Could not send the message");

    println!("Message sent to python initiating the sending process!");

    let mut buf = [0;60];
    let mut anomalies = Vec::<f32>::new();
    let mut regular = Vec::<f32>::new();
    for i in 0..3000{
        let (amt, _src) = socket.recv_from(& mut buf)?;
        if amt == 60 {
            let composite_id = u32::from_le_bytes(buf[0..4].try_into().unwrap());
            let float: Vec<f32> = buf[4..60]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
            
            let array = Array2::from_shape_vec((1, 14), float.clone())?;
            let outputs = session.run(inputs![TensorRef::from_array_view(&array)?])?;
            let reconstructed = outputs[0].try_extract_tensor::<f32>()?;

            let mse = array.iter()
            .zip(reconstructed.1.iter())
            .map(|(original, reconstructed)| (original - reconstructed).powi(2))
            .sum::<f32>() / (features_count as f32);

            let threshold = 0.05;
            if mse > threshold {
                anomalies.push(mse);
            }
            else {
                regular.push(composite_id as f32);
            }
            println!("{i}");
        }
    }
    // let end = start.elapsed();
    // println!("{:?}", end);
    println!("{:?}", anomalies);
    println!("Done, The number of anomalies detected: {}", anomalies.len());
    println!("The number of regular data points: {}", regular.len());

    Ok(())
}
