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

**Call 1 Motion Summary**:
A single concise, task-blind draft sentence describing the visible motion, interaction, and final persistent state from Overview Sheets and Keyframe Sheets. It is intermediate evidence and is not stored as the canonical Motion Summary.
_Avoid_: canonical conclusion, action plan, task progress

**Canonical Motion Summary**:
A single concise sentence returned by Call 2 after revising the Call 1 draft with the task context, high-resolution Boundary States, existing Boundary descriptions, and detail evidence. This is the Motion Summary stored in Canonical Evidence.
_Avoid_: raw Call 1 output, Task Role

**Action Description**:
A single task-conditioned sentence stating what the robot physically does during one Evidence Unit, grounded in the Canonical Motion Summary, Boundary States, and detail evidence.
_Avoid_: plan, intent, Task Role

**Task Role**:
A single concise sentence stating only what the Action Description contributes to the task without repeating the physical action or overstating progress.
_Avoid_: Description, Action Description, task success

**Atomic Action Claim**:
A key physical predicate in a Motion Summary or Action Description that must map to direct visual evidence from an appropriate camera view. Grasp, hold, release, and contact claims require supporting evidence from the corresponding wrist camera.
_Avoid_: plausible but unsupported substep, task-prior completion

**Canonical Evidence**:
The structurally validated evidence record committed to an Evidence Unit, distinct from provider attempts, semantic judgments, and debug artifacts.
_Avoid_: raw response, debug output

**Evidence Compilation**:
The task of producing canonical Boundary and Evidence Unit records without deciding whether the complete Behavior Document is semantically acceptable.
_Avoid_: quality audit, repair

**Targeted Repair**:
The sole automatic mutation path for canonical evidence after Sequence Audit identifies an issue. Issues sharing one owner Evidence Unit are repaired together within a bounded budget.
_Avoid_: Targeted Reprocessing, local causal retry, manual review

**Sequence Audit**:
The sole semantic document-level check that Boundary changes, persistent entities, transition descriptions, and Task Roles remain coherent.
_Avoid_: human review, task planning

**Document Gate**:
The all-or-nothing training eligibility decision applied after successful Sequence Audit and bounded Targeted Repair.
_Avoid_: Unit quality threshold, partial Guide

**Quarantined Document**:
A complete compiler output whose remaining evidence conflict could not be resolved within the automatic processing budget. It is retained for measurement but excluded from every default Reader and training path.
_Avoid_: needs review, partially accepted Guide
