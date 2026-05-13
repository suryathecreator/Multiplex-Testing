# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 45
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 0.5333 | 0.5556 | 0.6000 | 0.6667 | 0.8000 | 0.8222 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 6354.6 | 8559.6 | 12837.3 | 21970.0 | 43109.6 | 90258.2 |

## Detailed Matched-Per-k Summary

- shared_trace_group_8: pass@1 prompts=45, pass@2 prompts=45, pass@4 prompts=45, pass@8 prompts=45, pass@16 prompts=45, pass@32 prompts=45

## Retry Statistics

- shared_trace_group_8: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 5:12:06 (18725.9 seconds)
- started_at: 2026-05-04T12:31:04.620487+00:00
- finished_at: 2026-05-04T17:43:10.518641+00:00

## Matched Denominators

- pass@1: 45 matched prompts
- pass@2: 45 matched prompts
- pass@4: 45 matched prompts
- pass@8: 45 matched prompts
- pass@16: 45 matched prompts
- pass@32: 45 matched prompts
