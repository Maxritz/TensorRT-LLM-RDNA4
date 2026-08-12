# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone torch-free Vulkan inference server.

Bypasses tensorrt_llm/__init__.py entirely — no torch import, no C++ chain.
Pure Python + ctypes + numpy + Vulkan.

Provides:
  - /v1/chat/completions — OpenAI-compatible endpoint
  - /v1/models — model info
  - / — web UI for chatting

Usage:
    python vulkan_serve.py --model "E:\OLLAMA-Models\GGUF\acrux-500m-o1-journey-q6_k.gguf"
"""

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np

# Bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TLLM_VULKAN_BACKEND", "1")
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tensorrt_llm.vulkan_backend.numpy_bridge import VulkanDevice
from tensorrt_llm.vulkan_backend.weight_loader import ModelWeights, load_gguf, load_safetensors, load_safetensors_dir
from tensorrt_llm.vulkan_backend.forward_pass import KVCache, TransformerModel
from tensorrt_llm.vulkan_backend.vulkan_ops import sample_from_probs

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head><title>Vulkan Chat</title>
<style>
body { font-family: monospace; max-width: 800px; margin: 20px auto; padding: 20px; background: #1a1a2e; color: #e0e0e0; }
.message { margin: 10px 0; padding: 10px; border-radius: 5px; }
.user { background: #16213e; border-left: 3px solid #00d4ff; }
.assistant { background: #0f3460; border-left: 3px solid #e94560; }
#input { width: 80%; padding: 8px; background: #16213e; color: #e0e0e0; border: 1px solid #333; }
#send { padding: 8px 16px; background: #0f3460; color: #e0e0e0; border: none; cursor: pointer; }
#send:hover { background: #5000e8; }
.token-stream { font-size: 0.8em; color: #888; }
</style>
</head>
<body>
<h1>Vulkan LLM Chat</h1>
<div id="chat"></div>
<input type="text" id="input" placeholder="Type a message..." onkeydown="if(event.key==='Enter') sendMessage()">
<button id="send" onclick="sendMessage()">Send</button>
<div class="token-stream" id="stream"></div>
<script>
let conversation = [];
async function sendMessage() {
    const input = document.getElementById('input');
    const msg = input.value.trim();
    if (!msg) return;
    input.value = '';
    addMessage('user', msg);
    addMessage('assistant', '...');
    const resp = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
            model: 'vulkan-llm',
            messages: [{role:'user', content: msg}],
            max_tokens: 128,
            temperature: 0.8,
            top_p: 0.9,
            stream: false
        })
    });
    const data = await resp.json();
    const content = data.choices[0].message.content;
    document.querySelector('.assistant:last-child').innerHTML = escapeHtml(content);
}
function addMessage(role, content) {
    const chat = document.getElementById('chat');
    const div = document.createElement('div');
    div.className = 'message ' + role;
    div.innerHTML = '<b>' + role + ':</b> ' + escapeHtml(content);
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
</script>
</body>
</html>
"""


class VulkanServer:
    def __init__(self, model_path, max_tokens=128, temperature=0.8, top_k=50, top_p=0.9):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.device = None
        self.weights = None
        self.model = None
        self.kv_cache = None
        self._lock = threading.Lock()
        self._loaded = False
        self._tokenizer = None

    def _get_tokenizer(self):
        """Build a BPE tokenizer from GGUF vocab/merges using tiktoken."""
        if self._tokenizer is not None:
            return self._tokenizer
        from gguf import GGUFReader
        import tiktoken
        from tiktoken.data import load_tiktoken_bpe

        reader = GGUFReader(self.model_path, "r")
        tokens_field = reader.fields.get("tokenizer.ggml.tokens")
        if tokens_field is None:
            self._tokenizer = None
            return None

        # Extract vocab
        vocab = {}
        for i, idx in enumerate(tokens_field.data):
            val = tokens_field.parts[idx]
            if isinstance(val, (bytes, bytearray)):
                vocab[i] = val.decode("utf-8", errors="replace")
            else:
                vocab[i] = str(val)

        merges = []
        merges_field = reader.fields.get("tokenizer.ggml.merges")
        if merges_field:
            for i, idx in enumerate(merges_field.data):
                val = merges_field.parts[idx]
                if isinstance(val, (bytes, bytearray)):
                    merges.append(val.decode("utf-8", errors="replace"))
                else:
                    merges.append(str(val))

        # EOS/PAD/BOS tokens
        bos = 151643  # Qwen default
        eos = 151643
        pad = 151643
        try:
            eos = int(reader.fields["tokenizer.ggml.eos_token_id"].parts[0])
        except (KeyError, IndexError):
            pass
        try:
            bos = int(reader.fields["tokenizer.ggml.bos_token_id"].parts[0])
        except (KeyError, IndexError):
            pass

        # Build encoding using tiktoken's BPE
        enc = tiktoken.Encoding(
            name="gguf-bpe",
            pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            mergeable_bpe_ranks=vocab,
            special_tokens={
                "<|endoftext|>": eos,
                "<|im_start|>": 151643,
                "<|im_end|>": 151643,
                "<|fim_prefix|>": 0,
                "<|fim_middle|>": 0,
                "<|fim_suffix|>": 0,
                "<|fim_pad|>": 0,
                "<|fim_context|>": 0,
                "<|fim_st_pad|>": 0,
            },
        )
        self._tokenizer = (enc, vocab, eos, bos)
        return self._tokenizer

    def tokenize(self, text):
        """Tokenize text using BPE tokenizer from GGUF."""
        enc_vocab = self._get_tokenizer()
        if enc_vocab is None:
            return np.array([151643, 0, 0, 0, 0, 0], dtype=np.int32)

        enc, vocab, eos, bos = enc_vocab
        tokens = enc.encode(text, allowed_special=enc.special_tokens_set)
        if not tokens:
            return np.array([bos, 0, 0, 0, 0, 0], dtype=np.int32)
        return np.array(tokens, dtype=np.int32)

    def _bpe_tokenize(self, text, vocab, merges):
        """Simple BPE tokenization fallback."""
        return None

    def detokenize(self, token_ids):
        """Detokenize using GGUF vocab."""
        enc_vocab = self._get_tokenizer()
        if enc_vocab is None:
            vocab = {}
            try:
                from gguf import GGUFReader
                reader = GGUFReader(self.model_path, "r")
                tokens_field = reader.fields.get("tokenizer.ggml.tokens")
                if tokens_field:
                    for i, idx in enumerate(tokens_field.data):
                        val = tokens_field.parts[idx]
                        if isinstance(val, (bytes, bytearray)):
                            vocab[i] = val.decode("utf-8", errors="replace")
                        else:
                            vocab[i] = str(val)
            except Exception:
                pass

            result = ""
            for tid in token_ids:
                if tid in vocab:
                    result += vocab[tid]
                else:
                    result += f"<{tid}>"
            return result

        enc, vocab, _, _ = enc_vocab
        try:
            text = enc.decode(list(token_ids))
        except Exception:
            result = ""
            for tid in token_ids:
                if tid in vocab:
                    result += vocab[tid]
                else:
                    result += f"<{tid}>"
            return result
        return text
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            print("Initializing Vulkan device...")
            self.device = VulkanDevice()
            print("Loading model weights...")
            t0 = time.time()
            if self.model_path.endswith(".gguf"):
                self.weights = load_gguf(self.model_path, self.device)
            elif self.model_path.endswith(".safetensors"):
                self.weights = load_safetensors(self.model_path, self.device)
            elif self.model_path.endswith("/") or self.model_path.endswith("\\"):
                self.weights = load_safetensors_dir(self.model_path, self.device)
            else:
                raise ValueError(f"Unsupported model format: {self.model_path}")
            t_load = time.time() - t0
            print(f"Loaded {len(self.weights.vulkan_ptrs)} tensors in {t_load:.2f}s")
            print(f"  Layers: {self.weights.n_layers}, Hidden: {self.weights.hidden_dim}, "
                  f"Heads: {self.weights.n_heads}/{self.weights.n_kv_heads}, "
                  f"Vocab: {self.weights.vocab_size}")
            self.model = TransformerModel(self.weights)
            print(f"Model built ({self.weights.n_layers} transformer blocks)")
            self._loaded = True

    def generate(self, prompt_ids, max_new_tokens=None, temperature=None, top_k=None, top_p=None):
        if not self._loaded:
            self.load()
        
        max_new_tokens = max_new_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_k = top_k if top_k is not None else self.top_k
        top_p = top_p if top_p is not None else self.top_p

        with self._lock:
            kv_cache = KVCache(
                self.device, self.weights.n_layers, self.weights.n_kv_heads,
                self.weights.head_dim, max_seq_len=4096
            )
            
            # Prefetch prompt
            logits = self.model.forward(prompt_ids, position=0, kv_cache=kv_cache)
            
            generated = []
            position = len(prompt_ids)
            token = int(np.argmax(logits))
            
            for step in range(max_new_tokens):
                logits = self.model.forward(
                    np.array([token], dtype=np.int32), position=position, kv_cache=kv_cache
                )
                next_token = sample_from_probs(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p
                )
                generated.append(next_token)
                position += 1
                token = next_token
            
            kv_cache.free()
            return generated

    def tokenize(self, text):
        """Tokenize using sentencepiece model from GGUF."""
        # Try sentencepiece first (Qwen2 uses BPE/SentencePiece)
        try:
            from gguf import GGUFReader
            import sentencepiece as spm
            import tempfile, os

            reader = GGUFReader(self.model_path, "r")

            # Try to get the raw sentencepiece model
            model_field = reader.fields.get("tokenizer.ggml.model")
            if model_field:
                val = model_field.parts[list(model_field.data)[0]]
                model_bytes = bytes(val)
                # Try treating as raw sentencepiece model
                with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as f:
                    f.write(model_bytes)
                    tmp_path = f.name
                try:
                    sp = spm.SentencePieceProcessor()
                    sp.load(tmp_path)
                    tokens = sp.encode(text)
                    os.unlink(tmp_path)
                    if tokens:
                        return np.array(tokens, dtype=np.int32)
                except Exception:
                    os.unlink(tmp_path)

            # Try GGUF vocab field (for BPE tokenizer)
            vocab_field = reader.fields.get("tokenizer.ggml.tokens")
            if vocab_field:
                vocab = []
                for i, idx in enumerate(vocab_field.data):
                    val = vocab_field.parts[idx]
                    if isinstance(val, (bytes, bytearray)):
                        vocab.append(val.decode("utf-8", errors="replace"))
                    else:
                        vocab.append(str(val))

                # Get merges
                merges = []
                merges_field = reader.fields.get("tokenizer.ggml.merges")
                if merges_field:
                    for i, idx in enumerate(merges_field.data):
                        val = merges_field.parts[idx]
                        if isinstance(val, (bytes, bytearray)):
                            merges.append(val.decode("utf-8", errors="replace"))
                        else:
                            merges.append(str(val))

                # Simple BPE tokenization
                tokens = self._bpe_tokenize(text, vocab, merges)
                if tokens:
                    return np.array(tokens, dtype=np.int32)

        except Exception:
            pass

        # Fallback: use vocab lookup
        vocab = {}
        try:
            from gguf import GGUFReader
            reader = GGUFReader(self.model_path, "r")
            tokens_field = reader.fields.get("tokenizer.ggml.tokens")
            if tokens_field:
                for i, idx in enumerate(tokens_field.data):
                    val = tokens_field.parts[idx]
                    if isinstance(val, (bytes, bytearray)):
                        vocab[val.decode("utf-8", errors="replace")] = i
                    else:
                        vocab[str(val)] = i
        except Exception:
            pass

        # BOS token
        bos_token = 151643  # Qwen2 BOS default
        tokens = [bos_token]
        for word in text.split():
            if word in vocab:
                tokens.append(vocab[word])
            else:
                # Character fallback
                for ch in word:
                    key = f"<0x{ord(ch):02X}>"
                    if key in vocab:
                        tokens.append(vocab[key])
                    else:
                        tokens.append(0)

        return np.array(tokens, dtype=np.int32)

    def _bpe_tokenize(self, text, vocab, merges):
        """Simple BPE tokenization fallback."""
        return None

    def detokenize(self, token_ids):
        """Simple detokenizer"""
        vocab = {}
        try:
            from gguf import GGUFReader
            reader = GGUFReader(self.model_path, "r")
            tokens_field = reader.fields.get("tokenizer.ggml.tokens")
            if tokens_field:
                for i, idx in enumerate(tokens_field.data):
                    val = tokens_field.parts[idx]
                    if isinstance(val, (bytes, bytearray)):
                        vocab[i] = val.decode("utf-8", errors="replace")
                    else:
                        vocab[i] = str(val)
        except Exception:
            pass

        result = ""
        for tid in token_ids:
            if tid in vocab:
                result += vocab[tid]
            else:
                result += f"<{tid}>"
        return result

    def free(self):
        if self.kv_cache:
            self.kv_cache.free()
        if self.model:
            self.model.free()


_server = None


class RequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, content_type="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self._set_headers("text/html")
            self.wfile.write(HTML_TEMPLATE.encode())
        elif parsed.path == "/v1/models":
            self._set_headers()
            resp = {
                "data": [{
                    "id": "vulkan-llm",
                    "object": "model",
                    "owned_by": "vulkan-backend",
                    "size": _server.weights.vocab_size if _server and _server._loaded else 0,
                    "n_layers": _server.weights.n_layers if _server and _server._loaded else 0,
                    "hidden_dim": _server.weights.hidden_dim if _server and _server._loaded else 0,
                }],
                "object": "list"
            }
            self.wfile.write(json.dumps(resp).encode())
        elif parsed.path == "/health":
            self._set_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode())
        elif parsed.path.startswith("/static/"):
            self._set_headers("text/plain")
            self.wfile.write(b"Not found")
        else:
            self._set_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                params = json.loads(body)
                messages = params.get("messages", [])
                max_tokens = params.get("max_tokens", 128)
                temperature = params.get("temperature", 0.8)
                top_k = params.get("top_k", 50)
                top_p = params.get("top_p", 0.9)
                stream = params.get("stream", False)

                # Extract prompt from messages
                prompt = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        prompt += msg.get("content", "") + " "

                if not prompt.strip():
                    self._set_headers()
                    self.wfile.write(json.dumps({"error": "No prompt provided"}).encode())
                    return

                # Tokenize
                prompt_ids = _server.tokenize(prompt.strip())
                generated_ids = _server.generate(
                    prompt_ids, max_tokens, temperature, top_k, top_p
                )
                generated_text = _server.detokenize(generated_ids)

                response = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "vulkan-llm",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": generated_text},
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": len(generated_ids),
                        "total_tokens": len(prompt_ids) + len(generated_ids)
                    }
                }

                if stream:
                    self._set_headers("text/event-stream")
                    for i, token in enumerate(generated_ids):
                        chunk = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "vulkan-llm",
                            "choices": [{
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None
                            }]
                        }
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                    # Final chunk
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    self._set_headers()
                    self.wfile.write(json.dumps(response).encode())

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._set_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self._set_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    parser = argparse.ArgumentParser(description="Vulkan LLM Server")
    parser.add_argument("--model", required=True, help="Path to .gguf or .safetensors model")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--max-tokens", type=int, default=128, help="Default max new tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Default temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Default top-k")
    parser.add_argument("--top-p", type=float, default=0.9, help="Default top-p")
    parser.add_argument("--preload", action="store_true", help="Load model at startup")
    parser.add_argument("--prompt", type=str, default=None, help="Generate from prompt on startup")
    args = parser.parse_args()

    global _server
    _server = VulkanServer(
        model_path=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    if args.prompt:
        _server.load()
        tokens = _server.tokenize(args.prompt)
        print(f"Prompt: {args.prompt}")
        print(f"Tokens: {tokens}")
        generated = _server.generate(tokens)
        text = _server.detokenize(generated)
        print(f"Output: {text}")
        _server.free()
        return

    if args.preload:
        _server.load()

    server = HTTPServer((args.host, args.port), RequestHandler)
    print(f"\nServer running at http://{args.host}:{args.port}/")
    print(f"  OpenAI-compatible endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  Model info: http://{args.host}:{args.port}/v1/models")
    print(f"  Web UI: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        _server.free()
        server.shutdown()


if __name__ == "__main__":
    main()
