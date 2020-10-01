import socket
import json

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect(('127.0.0.1', 8080))
    s.sendall(b'')
    json_encoded = s.recv(1024)

    data = json.loads(json_encoded)

    print(data)
