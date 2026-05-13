# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 25
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 0.5200 | 0.5600 | 0.6800 | 0.7200 | 0.7200 | 0.7600 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 6897.4 | 9739.9 | 18224.1 | 36970.7 | 74636.8 | 151641.9 |

## Detailed Matched-Per-k Summary

- shared_trace_group_2: pass@1 prompts=25, pass@2 prompts=25, pass@4 prompts=25, pass@8 prompts=25, pass@16 prompts=25, pass@32 prompts=25

## Retry Statistics

- shared_trace_group_2: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 2:37:24 (9443.8 seconds)
- started_at: 2026-05-02T19:55:53.022773+00:00
- finished_at: 2026-05-02T22:33:16.863111+00:00

## Matched Denominators

- pass@1: 25 matched prompts
- pass@2: 25 matched prompts
- pass@4: 25 matched prompts
- pass@8: 25 matched prompts
- pass@16: 25 matched prompts
- pass@32: 25 matched prompts
