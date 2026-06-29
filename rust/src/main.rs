use std::net::UdpSocket;

fn main() -> std::io::Result<()> {
    let socket = UdpSocket::bind("127.0.0.1:0")?;

    let message = "Hey from Rust";

    let server_address  = "127.0.0.1:8008";

    socket.send_to(message.as_bytes(), server_address).expect("Could not send the message");

    println!("Message sent to python initiating the sending process!");

    let mut buf = [0;20];
    for _ in 0..2{
        let (amt, _src) = socket.recv_from(& mut buf)?;
        if amt == 20 {
            let float: Vec<f32> = buf
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

            println!("{:?}", float);
        }
    }

    Ok(())
}
