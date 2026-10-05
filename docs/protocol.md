# MeshTrain V1 Protocol

MeshTrain has two planes:

* **Control plane:** HTTP/JSON between workers or the CLI and the coordinator.
* **Data plane:** raw TCP between neighbouring pipeline stages, carrying binary `TensorPacket` frames.

> **Security:** V1 is for **trusted machines on a private network** only. The cluster token is a
> shared secret sent in clear text over HTTP and inside the TCP hello. There is no TLS and no per-peer
> authentication. Do not expose a coordinator or worker port to the Internet or to untrusted peers.

## 1. Data plane

### 1.1 Frame format

All integers are big-endian.

| offset | size | field | notes |
|---|---|---|---|
| 0 | 4 | magic | `b"MSHT"` |
| 4 | 1 | version | `1` |
| 5 | 1 | reserved | `0` |
| 6 | 4 | header_len | 1 … 65 536 |
| 10 | 8 | payload_len | ≤ receiver limit (`network.max_tensor_mb`, default 1 GiB) |
| 18 | header_len | header | UTF-8 JSON object |
| 18+H | payload_len | payload | raw C-contiguous tensor bytes |

### 1.2 Header (TensorPacket)

```json
{
  "tensor_id": "3f9c0a1b2c3d4e5f",
  "job_id": "tiny-transformer-20261005-130007-fe03",
  "step_id": 12,
  "microbatch_id": 3,
  "message_type": 1,
  "shape": [4, 32, 128],
  "dtype": "float32",
  "source_worker": "rtx8",
  "destination_worker": "",
  "meta": {"loss": 2.31}
}
```

| message_type | value | direction | payload |
|---|---|---|---|
| FORWARD_ACTIVATION | 1 | stage k → k+1 | activation `h_k` (detached) |
| BACKWARD_GRADIENT | 2 | stage k+1 → k | `dL/dh_k` (`x_{k+1}.grad`); `meta.loss` = microbatch loss, relayed to stage 0 |
| TARGET | 3 | stage 0 → … → last | labels for the microbatch (relayed unchanged by middle stages) |
| CONTROL | 4 | any | no tensor; `meta.command` (`HELLO`), or probe ops `PING`/`SYNC`/`BYE` |
| ACK | 5 | any | no tensor; probe replies |
| ERROR | 6 | any | no tensor; `meta.error` (the sender is aborting the job) |

`dtype` is one of `float32, float64, float16, bfloat16, int64, int32, int16, int8, uint8, bool`.

### 1.3 Validation (receiver side)

The receiver rejects the frame and closes the link with `ProtocolError` if any of these hold:

* bad magic, unknown version, header length 0 or above 64 KiB, or payload above the limit
* the header is not valid UTF-8 JSON, or not a JSON object
* `message_type` is unknown, or the ids are negative or not integers
* a string field is longer than 256 characters
* `shape` is not a list of at most 16 non-negative ints
* `dtype` is not in the list above
* `meta` is not a flat object of scalars
* `payload_len != prod(shape) * itemsize(dtype)`

Nothing received is ever unpickled, `eval`ed or imported. Tensors are rebuilt with
`torch.frombuffer` from the validated dtype and shape.

### 1.4 Connection setup

Every worker runs one data-plane listener (default port 29500). The connecting side sends a
`CONTROL` packet first:

```json
{"message_type": 4, "meta": {"command": "HELLO", "job_id": "...", "stage": 0, "token": "<cluster token>"}}
```

The listener rejects the connection if the token does not match. It then routes the connection by
its content:

* `job_id` + `stage`: parked until the stage executor for stage `stage + 1` claims its upstream
  link.
* `command_kind: "PROBE"`: handled by the network profiler.

Stage *k* connects to stage *k+1* (the address comes in the `START_STAGE` command). The single TCP
connection then carries traffic both ways.

### 1.5 Per-step exchange (GPipe, M microbatches)

```
stage 0                         stage 1 (middle)                  stage 2 (last)
for m in 0..M-1:
  TARGET(m)  ───────────────▶  relay  ──────────────────────────▶ buffer target m
  ACT(m)     ───────────────▶  forward, save ctx(step,m)
                                ACT(m) ─────────────────────────▶ forward+loss+backward
                                ◀───────────────────────────────── GRAD(m) {loss}
              ◀──────────────  backward ctx(step,m), GRAD(m) {loss}
backward ctx(step,m)
after M gradients: optimizer.step()  (each stage after its own M-th backward)
```

### 1.6 Timeouts and failure

* Each receive waits at most `network.timeout_s` for the next packet. A stalled frame also times out
  after `network.timeout_s`.
* A peer that disconnects is detected right away (EOF or reset).
* A stage that hits an exception sends `ERROR` to both neighbours before exiting, so the rest of the
  pipeline stops within milliseconds rather than waiting out the timeout.

### 1.7 Network probe

```
client → HELLO{command_kind: PROBE}
client → CONTROL{op: PING}   server → ACK        (× pings; RTT)
client → FORWARD_ACTIVATION (blob) × N            (no reply)
client → CONTROL{op: SYNC}   server → ACK{bytes}  (throughput = bytes / (elapsed − RTT))
client → CONTROL{op: BYE}
```

Each probe is directional. The coordinator measures i→j and j→i separately.

## 2. Control plane (HTTP)

Every endpoint except `/health` requires the `X-MeshTrain-Token` header. Requests larger than 8 MiB
are rejected with 413.

| method | path | body / params | response |
|---|---|---|---|
| POST | `/workers/register` | `{name, hardware, backend, device, data_host, data_port}` | `{worker_id, heartbeat_interval_s}` |
| POST | `/workers/{id}/heartbeat` | `{memory}` | `{ok}`; 404 means re-register |
| GET | `/workers/{id}/commands?timeout=s` | long poll, ≤ 30 s | `{commands: [...]}` |
| POST | `/workers/{id}/events` | `{type, data}` | `{ok}` |
| GET | `/cluster/status` | | `{workers, jobs}` |
| POST/GET | `/cluster/benchmark` | `{pings, payload_mb, network}` | progress / results |
| POST | `/plan` | `{config}` | `{plan, text}` (dry run) |
| POST | `/jobs` | `{config}` | job record + plan text |
| GET | `/jobs/{id}` | | status, losses, last metrics, summary |
| POST | `/jobs/{id}/stop` | | job record |

Worker `hardware` (from `profiler/hardware.py`):

```json
{"hostname": "m2-air", "platform": "macos", "cpu_count": 8, "ram_total": 17179869184,
 "accelerators": [{"backend": "mps", "name": "Apple arm64", "memory_total": 11453251584,
                   "unified_memory": true, "usable": true}],
 "capabilities": ["cpu", "float16", "float32", "float64", "mps", "unified_memory"]}
```

Commands sent from the coordinator to a worker:

| type | fields | worker reply event |
|---|---|---|
| `RUN_BENCHMARK` | `request_id` | `benchmark_result {request_id, result}` |
| `PROBE_NETWORK` | `request_id, target, host, port, pings, payload_mb` | `probe_result {request_id, result}` |
| `START_STAGE` | `job_id, config, stage_index, num_stages, layers, downstream{host,port}, upstream_worker` | `stage_metrics` per step, then `stage_done` or `stage_error` |
| `STOP_JOB` | `job_id, reason` | (the stage aborts) |

A failed command reports `command_error {request_id, error}`.

Liveness: a worker that sends no heartbeat (or command poll) for `heartbeat_timeout_s` (default
15 s) is marked `OFFLINE`. Any running job that uses that worker is marked `FAILED` with a message
naming it, and the surviving stages receive `STOP_JOB`.

## 3. Metrics record (`runs/<job>/metrics.jsonl`)

One `STEP_COMPLETE` record per stage per step:

```json
{"event": "STEP_COMPLETE", "job": "...", "worker": "rtx8", "stage": 0, "backend": "cuda", "step": 12,
 "step_s": 0.41, "forward_s": 0.08, "backward_s": 0.15, "optimizer_s": 0.01, "comm_s": 0.05, "idle_s": 0.12,
 "bytes_sent": 16200000, "bytes_received": 16100000, "utilization": 0.59,
 "lifecycle": {"detach": 1e-5, "to_cpu": 0.01, "serialize": 0.004, "network_send": 0.03,
               "deserialize": 0.002, "to_device": 0.006},
 "memory": {"parameters": 1, "gradients": 1, "optimizer_state": 2, "saved_activations": 0,
            "saved_activations_peak": 3, "device_allocated": 4, "device_peak": 5, "device_total": 6},
 "loss": 2.31, "samples_per_s": 39.0}
```

`loss` and `samples_per_s` appear only on stage 0, which receives every microbatch loss.
