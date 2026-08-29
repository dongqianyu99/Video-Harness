# Video Harness Documentation

Video Harness compiles successful RoboDojo demonstrations into source-grounded,
image-text Guidance Documents for guide-conditioned robot policies. Each source episode
produces one final Document, stored under its official RoboDojo task name:

```text
documents-openai/
└── put-bread-into-the-toaster/
    ├── episode-0000300.document.jsonl
    └── episode-0000301.document.jsonl
```

Start here:

- [User Guide](getting-started.md): environment setup, RoboDojo download, one-episode
  debug run, complete multi-worker processing, resume, and run summaries.
- [Architecture](architecture.md): temporal representation, quality gates, batch
  execution, failure recovery, and output contracts.
- [Evidence Protocol](evidence-prompt.md): the two-call VLM evidence contract, camera
  authority, and automatic repair behavior.

The canonical Pi0.5 interface is the task-grouped per-episode Document directory. The
aggregate JSONL produced by annotation is retained only for merge and reporting
compatibility.
