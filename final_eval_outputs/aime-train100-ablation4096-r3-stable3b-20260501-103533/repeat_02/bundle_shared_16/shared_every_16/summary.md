# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 100
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_16 | 0.4600 | 0.5300 | 0.5600 | 0.6000 | 0.6200 | 0.6700 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_16 | 6567.2 | 8967.7 | 13790.1 | 23451.0 | 42557.0 | 84633.7 |

## Detailed Matched-Per-k Summary

- shared_trace_group_16: pass@1 prompts=100, pass@2 prompts=100, pass@4 prompts=100, pass@8 prompts=100, pass@16 prompts=100, pass@32 prompts=100

## Retry Statistics

- shared_trace_group_16: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 2:23:35 (8614.6 seconds)
- started_at: 2026-05-02T08:25:36.107096+00:00
- finished_at: 2026-05-02T10:49:10.709224+00:00

## Matched Denominators

- pass@1: 100 matched prompts
- pass@2: 100 matched prompts
- pass@4: 100 matched prompts
- pass@8: 100 matched prompts
- pass@16: 100 matched prompts
- pass@32: 100 matched prompts
