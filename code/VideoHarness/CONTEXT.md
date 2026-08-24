# Video Harness Domain Language

Video Harness compiles one source demonstration into a source-grounded Behavior Document and derives Actuator-specific Guide Views without changing the canonical evidence.

## Language

**Evidence Unit**:
A fixed temporal transition connecting two adjacent Boundary States in one source episode. It is the atomic motion-analysis and transition-interpretation unit, not necessarily a complete task phase.
_Avoid_: Guidance Unit, task step, stage

**Boundary State**:
A synchronized three-view observation at one sampled episode frame, shared by the Evidence Units immediately before and after it. Its canonical static description exists once in the Behavior Document.
_Avoid_: endpoint copy, Unit before state, Unit after state

**Semantic Stage**:
A task-meaningful phase that may span one or more Evidence Units. It is an interpretation over temporal evidence rather than a fixed-duration sampling window.
_Avoid_: Unit, stage sheet

**Behavior Document**:
The canonical, source-grounded sequence of Evidence Units compiled from one complete source episode. It stores evidence text, frame references, source identity, and provenance rather than generated media.
_Avoid_: annotation file, caption log, Guide View

**Guide View**:
An Actuator-facing representation derived from a Behavior Document, such as an image-text-interleaved sequence. It may resolve referenced frames and render evidence text, but it does not modify the canonical document.
_Avoid_: Behavior Document, prompt file

**Overview Sheet**:
A dense chronological contact sheet used to inspect motion across most frames of one Evidence Unit.
_Avoid_: video summary

**Keyframe Sheet**:
A higher-resolution chronological sheet of sparsely sampled frames from one Evidence Unit. Its samples do not define Semantic Stages.
_Avoid_: stage sheet, stage view

**Motion Summary**:
The task-blind temporal account produced from Overview Sheets and Keyframe Sheets before task interpretation.
_Avoid_: action plan, task progress

**Canonical Evidence**:
The validated evidence record committed to an Evidence Unit, distinct from provider attempts and debug artifacts.
_Avoid_: raw response, debug output

**Targeted Reprocessing**:
An automatic evidence-directed recompilation invoked when the normal interpretation is inconsistent or insufficient. It is bounded and does not wait for human intervention.
_Avoid_: manual review, blind retry

**Sequence Audit**:
An automatic document-level check that every Boundary State change is explained by its neighboring Evidence Unit and that persistent interactions remain temporally coherent.
_Avoid_: human review, task planning

**Quarantined Document**:
A complete compiler output whose remaining evidence conflict could not be resolved within the automatic processing budget. It is retained for measurement but excluded from every default Reader and training path.
_Avoid_: needs review, partially accepted Guide
