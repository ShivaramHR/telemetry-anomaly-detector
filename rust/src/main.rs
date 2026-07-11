use std::net::UdpSocket;
use ndarray::Array2;
use ort::{inputs, session::Session};
use ort::value::TensorRef;
// use std::time::Instant;
use std::fs::OpenOptions;
use std::io::Write;

fn log_all_mse(composite_ids: &Vec<u32>, all_mse: &Vec<f32>) {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("test_mse_log.csv")
        .expect("Unable to open file");
    writeln!(file, "composite_id,mse").expect("Unable to write header to file");

    for (composite_id, mse) in composite_ids.iter().zip(all_mse.iter()) {
        writeln!(file, "{},{}", composite_id, mse).expect("Unable to write to file");
    }
}


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
    let mut all_mse = Vec::<f32>::new();
    let mut composite_ids = Vec::<u32>::new();
    for i in 0..2000{
        let (amt, _src) = socket.recv_from(& mut buf)?;
        if amt == 60 {
            let composite_id = u32::from_le_bytes(buf[0..4].try_into().unwrap());
            composite_ids.push(composite_id);
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

            all_mse.push(mse);
            let time_in_cycle = composite_id % 1000;

            let mut threshold = 0.04;
            if time_in_cycle < 20 {
                threshold = 0.05;
            }

            if mse > threshold {
                anomalies.push(mse);
            } 
            else {
                regular.push(mse);
            }
        }
        println!("{i}");
    }
    // let end = start.elapsed();
    // println!("{:?}", end);
    // println!("{:?}", anomalies);
    println!("Done, The number of anomalies detected: {}", anomalies.len());
    println!("The number of regular data points: {}", regular.len());
    // println!("{:?}", regular);
    println!("{:?}", all_mse);

    log_all_mse(&composite_ids, &all_mse);

    Ok(())
}
