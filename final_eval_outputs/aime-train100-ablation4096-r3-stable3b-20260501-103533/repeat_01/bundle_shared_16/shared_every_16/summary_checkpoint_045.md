# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 45
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_16 | 0.5333 | 0.5556 | 0.6000 | 0.6000 | 0.6667 | 0.7111 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_16 | 6288.9 | 8685.1 | 13035.3 | 21986.7 | 40305.0 | 82812.4 |

## Detailed Matched-Per-k Summary

- shared_trace_group_16: pass@1 prompts=45, pass@2 prompts=45, pass@4 prompts=45, pass@8 prompts=45, pass@16 prompts=45, pass@32 prompts=45

## Retry Statistics

- shared_trace_group_16: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:21:34 (1293.9 seconds)
- started_at: 2026-05-02T08:19:04.619416+00:00
- finished_at: 2026-05-02T08:40:38.551473+00:00

## Matched Denominators

- pass@1: 45 matched prompts
- pass@2: 45 matched prompts
- pass@4: 45 matched prompts
- pass@8: 45 matched prompts
- pass@16: 45 matched prompts
- pass@32: 45 matched prompts
