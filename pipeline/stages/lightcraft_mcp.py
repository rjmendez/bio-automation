"""Minimal LightCraft control-channel client: JSON-RPC over `lightcraft-cli mcp --connect`, one request in flight."""
import base64
import io
import json
import os
import subprocess

from PIL import Image

EXE = os.environ.get('LIGHTCRAFT_CLI', r'C:\Program Files\LightCraft\lightcraft-cli.exe')
PORT = os.environ.get('LIGHTCRAFT_CONTROL', '127.0.0.1:7980')
ROOT = os.environ.get('BIO_DATA', r'F:\bio-automation-data')


class Mcp:
    def __init__(self):
        self.p = subprocess.Popen([EXE, 'mcp', '--connect', PORT], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, encoding='utf-8', bufsize=1)
        self.n = 1
        self._s({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                 "clientInfo": {"name": "bio-pipeline", "version": "1"}}})
        self._r(1)
        self._s({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _s(self, o):
        self.p.stdin.write(json.dumps(o) + '\n')
        self.p.stdin.flush()

    def _r(self, i):
        while True:
            d = json.loads(self.p.stdout.readline())
            if d.get('id') == i:
                return d

    def call(self, name, args):
        self.n += 1
        self._s({"jsonrpc": "2.0", "id": self.n, "method": "tools/call", "params": {"name": name, "arguments": args}})
        r = self._r(self.n)
        return r.get('result', r)


def text(r):
    for c in r.get('content', []):
        if c.get('type') == 'text':
            try:
                return json.loads(c['text'])
            except Exception:
                return c['text']


def image(r):
    for c in r.get('content', []):
        if c.get('type') == 'image':
            return Image.open(io.BytesIO(base64.b64decode(c['data']))).convert('RGB')
