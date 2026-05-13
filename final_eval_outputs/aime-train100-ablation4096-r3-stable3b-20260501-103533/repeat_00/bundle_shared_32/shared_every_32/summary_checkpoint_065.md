# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 65
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 0.5385 | 0.6308 | 0.6615 | 0.7231 | 0.7385 | 0.7538 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 6430.1 | 8738.0 | 13346.7 | 22802.7 | 41424.0 | 78918.3 |

## Detailed Matched-Per-k Summary

- shared_trace_group_32: pass@1 prompts=65, pass@2 prompts=65, pass@4 prompts=65, pass@8 prompts=65, pass@16 prompts=65, pass@32 prompts=65

## Retry Statistics

- shared_trace_group_32: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:32:01 (1920.5 seconds)
- started_at: 2026-05-02T01:56:20.861154+00:00
- finished_at: 2026-05-02T02:28:21.373996+00:00

## Matched Denominators

- pass@1: 65 matched prompts
- pass@2: 65 matched prompts
- pass@4: 65 matched prompts
- pass@8: 65 matched prompts
- pass@16: 65 matched prompts
- pass@32: 65 matched prompts
