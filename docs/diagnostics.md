# Agent-friendly diagnostics

Swama writes local diagnostic events as UTF-8 JSON Lines to:

```text
~/.swama/logs/events.jsonl
```

Use `swama logs` to read the last 200 events or `swama logs --follow` to follow new events. Set
`SWAMA_DIAGNOSTICS_PATH` to choose another local file, or `SWAMA_DIAGNOSTICS_DISABLED=1` to disable
the stream. Swama keeps the current file and one rotated file, each bounded to 8 MiB; the read
command combines both.

The `swama logs` reader does not emit its own session records into the stream it is reading.

Diagnostics are separate from CLI stdout/stderr. They never contain prompts, generated text, tool
arguments, image/audio content, credentials, authorization headers, or raw filesystem paths.

## Envelope

Every line uses schema `swama.diag/1` and the following envelope:

| Field | Meaning |
| --- | --- |
| `schema` | Literal `swama.diag/1` |
| `ts` | UTC RFC3339 timestamp with milliseconds |
| `seq` | Per-session monotonic sequence starting at zero; a gap means records were dropped |
| `level` | `debug`, `info`, `warn`, or `error` |
| `subsystem` | Actual owner such as `session`, `model`, `generation`, or `diagnostics` |
| `event` | Event name from the closed set below |
| `session` | Process/run UUID |
| `op` | Operation UUID when the event belongs to a load, generation, or eviction |
| `model` | Logical model id; local paths are hashed |
| `duration_ms` | Terminal-event wall time only |
| `outcome` | Terminal-event result: `ok`, `error`, or `cancelled` |
| `error` | Error outcomes only: `{ "code": <bounded code>, "transient": <bool> }` |
| `data` | Event-specific object |

Malformed or truncated JSONL is `UNKNOWN`, never “no event.” File writes are serialized across
Swama processes. A primary-sink failure cannot fail inference; Swama emits a `log.dropped` record
to stderr as the fallback surface.

## Events

### Session

- `swama.session.started`: `data={pid, version, mode}` where mode is `app`, `serve`, or `cli`.
- `swama.session.stopped`: terminal event.

An active process has no stopped event yet. A process killed by a signal or crash may also lack the
terminal event; consumers must not invent a clean stop in that case.

### Model load

- `model.load.started`
- `model.load.completed`: terminal `outcome=ok`
- `model.load.cancelled`: terminal `outcome=cancelled`, with no error object
- `model.load.failed`: terminal `outcome=error`

Every load terminal carries:

```json
{
  "phases": {
    "config_read": null,
    "config_decode": null,
    "model_graph": null,
    "tokenizer": 138.4,
    "weights": null
  }
}
```

Each phase is `number | null`. `null` means the upstream public loader does not expose that
boundary; it does not mean zero or skipped. Non-null phases must be no greater than the terminal
`duration_ms`. Phases may overlap and must not be added together.

Clear, remove, or eviction can invalidate an in-flight load. Its only valid terminal is then
`model.load.cancelled`; a later `model.load.completed` for the invalidated operation is a stale-load
resurrection signal.

### Generation

- `generation.started`
- `generation.first_token`: `data={ttft_ms}`
- `generation.completed`: terminal `outcome=ok`, `data={input_tokens, output_tokens}`
- `generation.cancelled`: terminal `outcome=cancelled`, `data={output_tokens}`
- `generation.failed`: terminal `outcome=error`

### Model eviction

- `model.eviction.started`
- `model.eviction.completed`: terminal `outcome=ok`, `data={generation, resident_bytes_after}`
- `model.eviction.failed`: terminal `outcome=error`

### Tokenizer cache

These events are reserved for Swama's bounded tokenizer cache. They contain only a lowercase
SHA-256 fingerprint, never tokenizer content.

- `tokenizer.cache.hit`: `data={fingerprint}`
- `tokenizer.cache.miss`: `data={fingerprint}`
- `tokenizer.cache.evicted`: `data={fingerprint, reason}` where reason is `budget` or `explicit`
- `tokenizer.cache.rejected`: terminal `outcome=error`, `data={fingerprint}`

### Diagnostic sink

- `log.dropped`: `data={dropped_count}` on the fallback surface.

## Routing boundary

Swama records only engine facts: model lifecycle, durations, outcomes, and token counts. The host
agent owns routing and fallback. Swama does not calculate or emit `cloud_tokens_avoided`, because
it does not know the counterfactual cloud model or tokenizer. Routing events require explicit
caller-provided routing context and are not part of schema v1.
