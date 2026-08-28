# Working agreement for this repo

## progress.md is the memory of this project

`progress.md` is the living record: every phase, every decision and its rationale,
every deviation from the original design and why.

**Read it first.** At the start of any session in this repo, read `progress.md` before
doing anything else. It is the fastest way to recover full context.

**Update it before every commit or push.** No commit lands without `progress.md`
reflecting the work in it. Specifically, update:

- the **Status** table, when a phase changes state;
- the relevant **phase section**, with what was built and what was measured;
- **Decisions on record**, when a question is answered or an answer is revised;
- **Deviations from the original spec**, when reality diverges from the plan — always
  with the reason, never just the change;
- the **Last updated** line.

Record measured numbers, not estimates. If something was tried and abandoned, say so
and say why — the discarded paths are often the most useful part of the record.

## The system design artifact

The full design lives at
<https://claude.ai/code/artifact/7f2391b1-149f-4f3b-bbf0-37063d6d38dd>

Keep it in step with `progress.md`. When a design or scaffolding decision changes,
update the artifact too, and state the reason for the change rather than silently
editing the plan to match what was built.

## Project conventions

- **Phases are gated.** A phase is complete when its tests pass, not when the code runs.
- **Every claim is measured.** Match rates, coverage, correlations — run it, then state
  the number. No estimates presented as results.
- **Leakage is the standing risk.** The target is Transfermarkt value and one feature
  variant contains a prior Transfermarkt value. Any new feature must be checked for
  whether it could carry information unavailable at prediction time, and the check
  belongs in `tests/test_panel.py`.
- **Two variants, always reported together.** `coldstart` (no prior valuation) is the
  interesting model; `update` (with it) is the accurate one. Never quote one alone.
- **Baselines are not optional.** Last season's TM value carried forward is a brutally
  strong baseline. Every leaderboard reports it.
- Data files are build artefacts and stay out of git. `scripts/01_ingest.py` then
  `scripts/03_build_panel.py` regenerates everything in about a minute.
