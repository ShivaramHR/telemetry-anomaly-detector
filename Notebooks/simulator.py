import time
import struct 
import socket

HOST = "127.0.0.1"
PORT = 8008

server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server.bind((HOST, PORT))
print("Server is active")

#test dataset
dataset = [[1.0, 2.0, 3.0, 4.0, 5.0]]

packer = struct.Struct("<5f")

interval = 0.100
next_time = time.time()

try:
    _, address = server.recvfrom(1024)
    for row in dataset:     
        print("Sending _ bytes every 100 ms")
        
        next_time += interval

        byte = packer.pack(*row)

        server.sendto(byte, address)

        sleep_time = next_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
except KeyboardInterrupt:
    print("The user interrupted, Stopping the program!")
finally:
    server.close()
