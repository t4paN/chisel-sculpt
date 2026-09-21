# Chisel — Changes

Short, chronological log of notable changes. Newest on top.

## 2026-09-22 — Per-stage stroke timers, and what they refuted

*Measured at L10 (10,485,762 verts / 21M tris) on the Arc B570. Both backends build.*

`perf-L10-bottleneck-handoff.md` established that the GPU is **starved, not busy** — 87% of
its clock ceiling with work only 36% of the time — and named three CPU-side suspects
without evidence separating them. This adds the stopwatches, behind the existing
`CHISEL_DIRTY_HIST=1` flag, and the leading suspect turned out to be innocent.

Four stages, timed inside the functions rather than at the call sites, because
`snap_and_mirror_dirty` and `post_dab` are each reached from several places and timing one
site would have missed the rest:

| Stage | Share of stroke wall |
|---|---|
| `take` — the readback copy | **1%** |
| `snap` — undo snapshot + mirror twins | **1%** |
| `expand` — dirty → affected adjacency | **17–29%** |
| `normals` — sort/unique + dispatch | **20–31%** |

**The readback is not the bottleneck.** The prior handoff put it first: ~19 MB per dab, 60x
the rescue-era figure, "roughly 570 MB of avoidable PCIe traffic per stroke", with
tightening `kCapSlack` as the suggested lever. It costs **0.5 ms a dab, 1% of the stroke**.
Optimising it would have bought nothing measurable. The 800K-id snapshot walk is equally
innocent at 1%.

The cost is in the two normal-related stages. One big-brush stroke spent 675 ms in `expand`
and 703 ms in `normals` across six dabs — **176 ms average for a single normals pass, 422 ms
at worst**.

The report prints a stroke wall-clock and an explicit **unaccounted** line, which is what
makes it a detector rather than four numbers: at 39–88% unaccounted, these four stages are
demonstrably not the whole story, and the next probe belongs somewhere else. Wall clock
only — no GPU sync, nothing that perturbs what it measures.

### Two things the same run exposed, both still open

**`expand` runs exactly twice per dab** in every stroke measured (50 calls/25 dabs,
84/42, 124/62, 12/6). There are two `post_dab` sites: one fires synchronously in
`src/main.cpp` right after dispatch, the other when the async dirty list lands. At the
first, the dab's own list has not arrived, so `dirty_verts` still holds the *previous*
dab's contents. Whether that call is redundant or is deliberately keeping normals current
while the real list is in flight has **not** been established — worth ~8–15% of stroke time
if it is waste.

**Every big-brush dab at L10 overflows its arena region, by arithmetic.** Six
`[arena] dab overflowed` events, five of them into a 2,621,440-id region — which is not an
estimate but `est / 4`, the last-ditch fallback in `begin_dab`. A worst-case region at L10
needs 40 MB and must be *contiguous*, so it only fits when the ring is empty and aligned;
with dabs pipelining it never is, both allocation attempts fail, and the quarter-size
fallback is what remains. A big-brush dab genuinely needs 3.4–4.2M ids — it touches 33–40%
of the whole mesh — so 2.6M can never satisfy it. This is the likely source of the undo
exhaustion first seen at L10: each overflow forces a snapshot over all 10.5M verts.

The first overflow is a separate bug: its 6,800-id region came from small-brush history.
The rescale meant to catch a brush-size change reads `anchor_world_radius`, but `begin_dab`
runs *before* this dab's anchor is set, so on a stroke's first dab it is still the previous
stroke's radius and no correction happens.

Neither is fixed here. Both have obvious-looking one-line candidates, and that is precisely
the shape of thing this arc has twice gotten wrong by reasoning instead of measuring.

