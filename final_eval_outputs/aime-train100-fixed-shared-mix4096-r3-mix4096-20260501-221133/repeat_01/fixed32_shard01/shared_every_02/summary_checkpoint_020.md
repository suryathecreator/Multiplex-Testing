# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 20
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 0.4500 | 0.5000 | 0.6500 | 0.7000 | 0.7000 | 0.7000 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 7021.9 | 10030.5 | 18987.5 | 37848.1 | 76370.5 | 155404.5 |

## Detailed Matched-Per-k Summary

- shared_trace_group_2: pass@1 prompts=20, pass@2 prompts=20, pass@4 prompts=20, pass@8 prompts=20, pass@16 prompts=20, pass@32 prompts=20

## Retry Statistics

- shared_trace_group_2: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 2:37:17 (9436.7 seconds)
- started_at: 2026-05-02T19:55:53.022773+00:00
- finished_at: 2026-05-02T22:33:09.741431+00:00

## Matched Denominators

- pass@1: 20 matched prompts
- pass@2: 20 matched prompts
- pass@4: 20 matched prompts
- pass@8: 20 matched prompts
- pass@16: 20 matched prompts
- pass@32: 20 matched prompts
