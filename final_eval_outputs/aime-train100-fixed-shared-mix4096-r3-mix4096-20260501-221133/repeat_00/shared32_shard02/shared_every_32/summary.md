# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 25
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 0.4800 | 0.5200 | 0.5200 | 0.5200 | 0.5600 | 0.6000 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 6802.3 | 9251.6 | 14343.6 | 24388.5 | 44375.9 | 83786.7 |

## Detailed Matched-Per-k Summary

- shared_trace_group_32: pass@1 prompts=25, pass@2 prompts=25, pass@4 prompts=25, pass@8 prompts=25, pass@16 prompts=25, pass@32 prompts=25

## Retry Statistics

- shared_trace_group_32: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:34:39 (2078.5 seconds)
- started_at: 2026-05-02T09:52:30.389631+00:00
- finished_at: 2026-05-02T10:27:08.911843+00:00

## Matched Denominators

- pass@1: 25 matched prompts
- pass@2: 25 matched prompts
- pass@4: 25 matched prompts
- pass@8: 25 matched prompts
- pass@16: 25 matched prompts
- pass@32: 25 matched prompts
