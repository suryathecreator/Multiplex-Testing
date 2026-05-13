# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 10
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 0.9000 | 0.9000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 5575.6 | 7177.2 | 14476.4 | 29349.9 | 55363.8 | 114597.1 |

## Detailed Matched-Per-k Summary

- shared_trace_group_2: pass@1 prompts=10, pass@2 prompts=10, pass@4 prompts=10, pass@8 prompts=10, pass@16 prompts=10, pass@32 prompts=10

## Retry Statistics

- shared_trace_group_2: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 4:17:13 (15432.7 seconds)
- started_at: 2026-05-02T18:41:19.402720+00:00
- finished_at: 2026-05-02T22:58:32.141671+00:00

## Matched Denominators

- pass@1: 10 matched prompts
- pass@2: 10 matched prompts
- pass@4: 10 matched prompts
- pass@8: 10 matched prompts
- pass@16: 10 matched prompts
- pass@32: 10 matched prompts
