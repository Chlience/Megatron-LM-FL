# Reviewing MegaLens Core integration

MegaLens observation helpers live in `megatron/megalens/probes/`. Core retains
the native computation, collective launches, waits and scheduling decisions,
with observation calls at the boundaries that define each event. Paired
`BEGIN/END MEGALENS OBSERVABILITY` comments identify those integration points.

## Implementation ownership

| Native entry point | Observation implementation | Boundary retained in the native flow |
|---|---|---|
| `core/pipeline_parallel/p2p_communication.py` | `megalens/probes/p2p.py` | Launch, raw Work registration, wait and existing synchronization |
| `core/pipeline_parallel/schedules.py` | `megalens/probes/pipeline.py` | Forward/backward phases, workload FIFO and gradient synchronization |
| `core/tensor_parallel/mappings.py` | `megalens/probes/tp.py` | Original equal/unequal collective branches and tensor transformations |
| `training/training.py` | `megalens/probes/training.py` | Runtime ownership and the iteration envelope containing skip/continue/break |
| DP and MoE entry points | `megalens/probes/dp.py`, `megalens/probes/moe.py` | Existing coalescing, stream, handle and postprocess boundaries |

Paths in the table are relative to `megatron/`. The generic
`core/observability.py` sink/gate/scope facade remains backend independent.
Existing TP/MoE observation imports remain compatibility entry points.

P2P keeps main's explicit even/odd rank branches and `_p2p_ops` signature. Its
small `_p2p_groups` helper shares main's native PP/WORLD group selection with
enabled observation, so recorded backend fields describe the actual request
group. Observation resolves this metadata lazily. This helper adds no group
creation, collective or synchronization.

Schedule metadata builders and training lifecycle helpers stay in the probe
package. Their callers retain the local scopes needed to observe failures and
early exits. The mode-1 iteration barrier stays before scope entry; skipped
iterations close the existing trace window. Tensor-parallel helpers preserve
the single-rank AllReduce scope and each event's existing group-slot timing.

## Invariants when updating upstream

- Preserve native signatures, branch order, return values, exceptions and raw
  Work objects. Keep collective arguments and wait/synchronize counts unchanged.
- Keep contiguous/reshape operations and stream/coalescing contexts within their
  existing observation boundaries. Moving a scope changes the measured event.
- A closed gate skips metadata and peer construction. Core imports must not load
  the Trace runtime, Tracer, analysis modules or optional collectors.
- Retain local `with` blocks around native work. Extract reusable observation
  logic without moving native training functions into the probe package.
- Maintain event names, fields and lifecycle semantics in
  `tests/megalens/fixtures/probe_scan_gate.json` and the event catalog.

## Verification

From the repository root in a configured development environment, run the CPU
gate against MixedPara revision `12fb7169ce30fdb62b50f86b41afa09336a523ea`:

```bash
rtk proxy python -B tests/test_utils/runners/run_megalens_gate.py \
  --profile cpu --source-repo ../MixedPara --timeout 300
```

Use `--source-repo` to point at that checkout. The scanner
checks the source/target event contract; the CPU suites exercise disabled gates,
field construction, raw Work identity, exceptions and import boundaries.
The focused suites are `test_p2p_observability.py`,
`test_pipeline_schedule_observability.py`, `test_runtime_lifecycle.py` and the
three `test_tp_*_observability.py` suites under `tests/megalens/`.

For a structural refactor, compare moved definitions and preserved native
functions by AST against the fixed source revisions. Compare P2P after inlining
only native group selection, and TP mappings after removing observation scope
wrappers. Build sdist/wheel and run `tests/megalens/wheel_import_check.py` in an
environment without the source checkout to verify the relocated modules ship.

CPU and AST checks do not establish GPU completion or communication timing.
Recheck affected trace-off/on training profiles and checkpoints on the actual
backend. `tests/test_utils/runners/run_p2p_order_nccl.py` verifies two-rank NCCL
launch order, request groups, payloads and raw Work identity. Performance overhead
requires a separate controlled measurement.

The [related merge requests](megalens-related-mrs.md) document owns independent
training/dependency fixes and their merge dependencies.

The [validation record](megalens-core-validation.json) fixes the reviewed code
revisions and records the source comparisons, CPU gate, packaging, installed
module origins and focused H800 training/NCCL results, with their evidence limits.
