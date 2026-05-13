# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 25
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 0.3200 | 0.3600 | 0.4000 | 0.4000 | 0.5200 | 0.5200 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 6869.2 | 9793.7 | 15514.6 | 27028.9 | 49517.2 | 94921.8 |

## Detailed Matched-Per-k Summary

- shared_trace_group_32: pass@1 prompts=25, pass@2 prompts=25, pass@4 prompts=25, pass@8 prompts=25, pass@16 prompts=25, pass@32 prompts=25

## Retry Statistics

- shared_trace_group_32: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:31:06 (1865.9 seconds)
- started_at: 2026-05-02T10:24:21.572637+00:00
- finished_at: 2026-05-02T10:55:27.514903+00:00

## Matched Denominators

- pass@1: 25 matched prompts
- pass@2: 25 matched prompts
- pass@4: 25 matched prompts
- pass@8: 25 matched prompts
- pass@16: 25 matched prompts
- pass@32: 25 matched prompts
