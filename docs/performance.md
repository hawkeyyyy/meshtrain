# Performance: overlap, buffers, CUDA streams

## Measuring

Every phase of every microbatch is an interval span: compute, D2H, serialize, queue wait, network send and
receive, deserialize, H2D, and waits. Per stage and step:

```
compute_ms               = |∪ compute spans|
communication_ms         = |∪ transfer spans (any thread)|      (queue wait excluded: no work)
overlapped_ms            = |compute ∩ communication|
exposed_communication_ms = communication_ms − overlapped_ms
idle_ms                  = step − |compute ∪ communication|
overlap_ratio            = overlapped_ms / communication_ms
```

Outputs:
* `runs/<id>/timeline.json`: Chrome Trace Event format; open in chrome://tracing or https://ui.perfetto.dev.
* `timeline_report.txt`: per-worker percentages and an ASCII Gantt chart of one step.
* `prediction_accuracy.json`: planner predictions vs measurements.

## Comparing modes

```sh
meshtrain benchmark pipeline                              # local processes, emulated 100 Mbit/s link
meshtrain benchmark pipeline --bandwidth-mbps 0           # raw loopback
meshtrain benchmark pipeline --devices cuda,cuda          # local GPUs
meshtrain benchmark pipeline --cluster --config configs/two_cuda.yaml   # real cluster
```

The results go into `docs/v1-vs-v1.5.md`. **Emulated links** (`networking/emulation.py`) throttle each direction
to a bandwidth and latency. They exist only because loopback hides every network effect on a single host.
Results produced with them always say so.

## Send path

1. `begin_d2h` runs on the compute thread. On CPU it is a zero-copy reference. On CUDA it records an event on
   the compute stream. On MPS it is a synchronous copy.
2. The item is queued in a bounded per-link FIFO.
3. On the sender thread, the D2H copy finishes. On CUDA: the transfer stream waits for the event, copies into
   a **pinned** pool buffer, then waits for that copy only.
4. The payload is sent zero-copy from tensor memory (header and payload are written separately).
5. The pinned buffer returns to the pool.

## Receive path

1. The reader thread validates the header and receives the payload **in place** into a pool buffer of the
   right shape and dtype (`Transport.payload_allocator`). There are no intermediate `bytes` copies.
2. On arrival, the compute thread calls `begin_h2d`. On CUDA this starts a non-blocking copy on the transfer
   stream and records an event. On CPU it is the buffer itself. On MPS it is a synchronous copy.
3. Just before use, `finish_h2d` runs. On CUDA the compute stream `wait_event`s, and `record_stream` protects
   the tensor from early reuse.
4. Buffers return to the pool:
   * CUDA host buffers once their copy event completes;
   * CPU buffers (which *are* the activation) only after that microbatch's backward.

After warm-up a stage allocates no new receive buffers (tested). Pool statistics (`allocation_count`,
`reuse_count`, `bytes_reserved`, pinned fallbacks) appear in every step's metrics as `buffer_pool`.

## CUDA streams: what overlaps and where it synchronises

| Copy | Stream | Ordered after | Waited for by |
|---|---|---|---|
| D2H of an activation or gradient | transfer | compute-stream event recorded at submit | the sender thread (`done.synchronize()`), not the compute thread |
| H2D of a received tensor | transfer | — | the compute stream (`wait_event`) just before the forward/backward that uses it |

* Forward and backward timing waits only for the **compute stream** (`sync_compute`), not the whole device.
  `torch.cuda.synchronize()` would also wait for transfers and silently serialise everything.
* Pinned buffers are reused only after their copy's event completes (`BufferPool.release(buf, event)`).
* Limitations:
  * GPU kernel times are host-measured around a compute-stream sync; there are no CUDA-event kernel timers yet.
  * Copies and kernels contend for PCIe and memory bandwidth.
  * Laptop GPUs and Windows WDDM may serialise more than the stream model suggests.
  * The CUDA code paths are covered by `@pytest.mark.cuda` tests that **have not run in the development
    environment** (no GPU). Run `uv run pytest -m cuda` on a CUDA machine.

## Results so far

Local CPU, 3 stages, tiny Transformer (5.4M parameters), batch 32 = 8 microbatches, emulated 100 Mbit/s and
1 ms per message:

| | V1 (GPipe, blocking) | V1.5 (1F1B, async) |
|---|---|---|
| step time | 864 ms | 501 ms (1.72x throughput) |
| exposed communication per stage | 247 ms | 112 ms |
| stage-0 saved activations | 37.9 MB | 14.2 MB |
| loss trajectory | — | identical |

GPipe with async sends reached about the same throughput as 1F1B with async sends in this setting. The 1F1B
gain is memory (2.7x less on stage 0), not speed. Full table: `docs/v1-vs-v1.5.md`. No real-network or GPU
numbers have been recorded for V1.5 yet.
