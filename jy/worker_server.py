#!/usr/bin/env python3
"""Serve `canvas_worker MODEL tail` on a loopback TCP port, one worker per connection.

The tunnel (dumbpipe listen-tcp) forwards each remote stream here, so the driver's
stdio protocol reaches the worker unchanged. Loopback only: the tunnel is the
authenticated hop.
"""
import os
import socket
import subprocess
import sys

port, root, model = int(sys.argv[1]), sys.argv[2], sys.argv[3]
server = socket.create_server(('127.0.0.1', port), reuse_port=False)
print(f'worker server on 127.0.0.1:{port}', flush=True)
while True:
    conn, _ = server.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    home = os.path.join(root, 'home')
    os.makedirs(home, exist_ok=True)
    print('connection: starting tail worker', flush=True)
    worker = subprocess.Popen([os.path.join(root, 'canvas_worker'), model, 'tail'], cwd=root,
                              env=dict(os.environ, HOME=home), stdin=conn.fileno(),
                              stdout=conn.fileno(), stderr=open(os.path.join(root, 'canvas-tail.log'), 'ab'))
    conn.close()
    print('tail worker exited', worker.wait(), flush=True)
