#!/usr/bin/env python3.10
"""
Embedding 常驻服务 —— 预加载 MiniLM 模型，通过 HTTP 提供 embedding + 覆盖检查。
启动: python3.10 embedding_server.py --port 5199
内存: ~1GB RSS（模型 420MB 磁盘 + Python/transformers 开销）
"""

import json
import sys
import os
import time
import signal
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import numpy as np

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model = None
_start_time = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[embedding-server] model loaded in {time.time()-t0:.1f}s", flush=True)
    return _model

class EmbeddingHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # 静默，避免污染 hermes 日志

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json({
                "status": "ok",
                "model": MODEL_NAME,
                "uptime_s": time.time() - _start_time if _start_time else 0,
                "memory_mb": _get_memory_mb(),
            })
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json"}, 400)
            return

        if path == "/embed":
            texts = data.get("texts", [])
            if not texts:
                self._send_json({"error": "missing texts"}, 400)
                return
            model = get_model()
            embeddings = model.encode(texts).tolist()
            self._send_json({"embeddings": embeddings, "dim": len(embeddings[0]) if embeddings else 0})

        elif path == "/check":
            items = data.get("items", [])
            reply = data.get("reply", "")
            threshold = data.get("threshold", 0.18)
            if not items or not reply:
                self._send_json({"error": "missing items/reply"}, 400)
                return

            model = get_model()
            emb_items = model.encode(items)
            emb_reply = model.encode(reply)

            results = []
            for i, (item, emb_i) in enumerate(zip(items, emb_items), 1):
                sim = float(np.dot(emb_i, emb_reply) /
                           (np.linalg.norm(emb_i) * np.linalg.norm(emb_reply)))
                results.append({"index": i, "item": item, "similarity": round(sim, 4),
                               "covered": sim >= threshold})
            self._send_json({"results": results, "threshold": threshold})

        elif path == "/shutdown":
            self._send_json({"status": "shutting down"})
            # 延迟关闭让响应先发出
            def _shutdown():
                time.sleep(0.1)
                os._exit(0)
            import threading
            threading.Thread(target=_shutdown, daemon=True).start()

        else:
            self._send_json({"error": "not found"}, 404)

def _get_memory_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1024 / 1024)
    except:
        return 0

def main():
    global _start_time
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5199

    print(f"[embedding-server] loading model: {MODEL_NAME}", flush=True)
    _start_time = time.time()
    get_model()  # 预加载
    mem = _get_memory_mb()
    print(f"[embedding-server] ready on port {port}, memory: {mem}MB RSS", flush=True)

    server = HTTPServer(("127.0.0.1", port), EmbeddingHandler)
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    server.serve_forever()

if __name__ == "__main__":
    main()
