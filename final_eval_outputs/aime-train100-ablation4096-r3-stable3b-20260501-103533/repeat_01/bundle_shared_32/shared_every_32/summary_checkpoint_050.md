# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 50
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 0.6000 | 0.6400 | 0.6800 | 0.7000 | 0.7000 | 0.7400 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 6088.9 | 8124.3 | 12383.5 | 20576.3 | 36835.3 | 69466.7 |

## Detailed Matched-Per-k Summary

- shared_trace_group_32: pass@1 prompts=50, pass@2 prompts=50, pass@4 prompts=50, pass@8 prompts=50, pass@16 prompts=50, pass@32 prompts=50

## Retry Statistics

- shared_trace_group_32: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 1:12:09 (4329.2 seconds)
- started_at: 2026-05-02T01:24:15.235665+00:00
- finished_at: 2026-05-02T02:36:24.449743+00:00

## Matched Denominators

- pass@1: 50 matched prompts
- pass@2: 50 matched prompts
- pass@4: 50 matched prompts
- pass@8: 50 matched prompts
- pass@16: 50 matched prompts
- pass@32: 50 matched prompts
