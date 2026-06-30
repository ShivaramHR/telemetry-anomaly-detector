import time
import struct 
import socket
import pandas as pd
import numpy as np

HOST = "127.0.0.1"
PORT = 8008

server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server.bind((HOST, PORT))
print("Server is active")

#test dataset
features = np.load("/Users/shivaram/telemetry-anomaly-detector/Data/Test_Data/scaled_features.npy") 
identifier = np.load("/Users/shivaram/telemetry-anomaly-detector/Data/Test_Data/identifiers.npy")


interval = 0.100
next_time = time.time()

try:
    _, address = server.recvfrom(1024)
    for i in range(0, 2):     
        next_time += interval

        id = int(identifier[i])
        feat = features[i]

        packet = struct.pack("<I14f", id, *feat)

        server.sendto(packet, address)

        sleep_time = next_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
except KeyboardInterrupt:
    print("The user interrupted, Stopping the program!")
finally:
    server.close()
