# ARCHON Thesis Presentation - Single-Tone Speaker Notes

Use this as a natural, first-person script in one consistent voice.

---

## Slide 1 - Title Slide

- Hi all, good morning. My name is Mansoor, my ID is 2024DA04102, and I am from the WILP MTech programme.
- My project title is ARCHON - Arch-Aware Context Network for Dental Enumeration and Diagnosis on Panoramic X-rays.
- ARCHON means the model understands the overall dental arch context, not just isolated teeth.
- I would also like to acknowledge my supervisor Priyank Bhardwaj, evaluator C V Krishnaveni, and examiner Murali N.
- This is my final review submission, July 2026.
- I will start with the clinical problem this work solves.

---

## Slide 2 - Problem Statement

- Let me first frame the problem.
- Panoramic dental X-rays are common in practice, but interpretation is still time-consuming and error-prone.
- Most existing AI systems do either tooth detection or disease detection, but not both together.
- They also usually do not provide severity grading in one pass.
- Symmetric teeth such as FDI 16 and 26 are often swapped across the midline.
- And most models do not learn full-arch spatial relationships.
- ARCHON addresses all of these gaps in one unified pipeline.
- Next, I will quickly explain the dataset split we used.

---

## Slide 3 - DENTEX Dataset Segments

- This work is based on the DENTEX 2023 dataset.
- Part 1 has around 300 images with bounding boxes and quadrant labels.
- Part 2 has around 500 images with full 32-class FDI mapping.
- Part 3 has around 400 images with disease attributes: caries, deep caries, impaction, and periapical lesion.
- There are also around 300 unlabelled images.
- In Phase 1, I train detection on all labelled data.
- Then I generate pseudo-labels on unlabelled images and expand training data for Phase 2.
- In Phase 2b and Phase 3, I focus more on disease attributes and severity.
- For this run, the scope is 1500 train, 50 val, 250 test, and 700 images in GT validation.
- Now I will walk through the architecture.

---

## Slide 4 - Architecture (Diagram)

- This slide is visual, so I will point to the diagram.
- ARCHON has five core improvements over the YOLOrtho baseline.
- These are CoordConv, Swin-based global context, cross-attention fusion, hybrid multi-task head, and quadrant-consistency penalty.
- It is a single end-to-end model, not an ensemble.
- Next, I will explain the training strategy first.

---

## Slide 5 - Training Phases (Diagram)

- I designed the training as a curriculum because DENTEX has hierarchical annotations.
- Each phase builds on the previous phase.
- The main rule is to improve attributes and severity without hurting detection.
- So in later phases, I freeze the backbone to prevent mAP regression.
- Before the phase details, I will explain the total loss.

---

## Slide 6 - Total Loss

- The total loss is a weighted sum of multiple objectives.
- GIoU loss checks how well the model localizes tooth boxes.
- Classification loss checks whether the model predicts the correct FDI class out of 32.
- DFL refines boundary precision.
- Disease BCE loss handles disease presence and absence.
- So the model learns detection, localization, identification, and disease prediction together.
- In Phase 3, I add severity and quadrant consistency terms.
- Now I will compare baseline and ARCHON design.

---

## Slide 7 - Baseline (YOLOrtho) vs ARCHON

- ARCHON keeps the strong parts of YOLOrtho: YOLOv8x plus CoordConv backbone and FPN.
- I add GlobalContextEncoder, MultiScaleFusion, HybridMultiTaskHead, CLAHE, and quadrant penalty.
- In one line, ARCHON is YOLOrtho plus global arch awareness plus severity grading.
- Next, I will show the full pipeline flow.

---

## Slide 8 - ARCHON Architecture Overview

- Input is a 1280x640 panoramic X-ray.
- Backbone is YOLOv8x plus CoordConv generating P3, P4, and P5 features.
- Swin Transformer on P5 captures full-arch context.
- Cross-attention fuses this global context back into all scales.
- Hybrid head outputs bounding box, FDI class, and disease severity signals.
- Post-processing uses locality-sensitive assignment and quadrant consistency to produce final JSON.
- Feature scales are P3 320 channels stride 8, P4 640 channels stride 16, P5 640 channels stride 32.
- Next, I will zoom into the backbone design.

---

## Slide 9 - Backbone: YOLOv8x + CoordConv + Modified FPN

- I selected YOLOv8x because 32-class FDI mapping needs high representational capacity.
- It has around 68.5M parameters and fits on a 12GB GPU with batch size 4.
- CoordConv adds normalized x and y channels so the network knows position.
- This is critical because FDI is position-dependent.
- I also use a modified FPN with strides 4, 8, and 16 to improve incisor detail.
- Forward hooks at layers 15, 18, and 21 extract features without altering Ultralytics internals.
- Next, I will explain the global context encoder.

---

## Slide 10 - Improvement A: GlobalContextEncoder (Swin)

- The issue is that CNN receptive field on P5 sees only a limited portion of the panoramic width.
- I solve this with a lightweight two-block Swin Transformer on P5.
- Block 1 does window attention; Block 2 does shifted window attention for cross-window communication.
- This gives near-global context at much lower cost than full self-attention.
- The added trainable parameters are around 3.2M out of roughly 79.5M total.
- The key point is that the model can learn tooth-order relationships across the arch.
- Next, I fuse this context into all scales.

---

## Slide 11 - Improvement B: MultiScaleFusion (Cross-Attention)

- At each scale, query comes from CNN local features, and key/value come from Swin global features.
- At P5, this helps quadrant disambiguation.
- At P4, this helps disease-type context.
- At P3, this helps fine lesion localization.
- I keep a residual path, so baseline CNN information is always preserved.
- Attention follows softmax(QK^T/sqrt(d_k))V.
- Next, I will explain the hybrid multi-task head and severity output.

---

## Slide 12 - Improvement C: HybridMultiTaskHead - Severity

- This is one key differentiator of ARCHON.
- ARCHON outputs clinically usable severity information, not only disease presence.
- I use a three-level interpretation where relevant: healthy, mild, and severe.
- Caries uses three levels.
- Impaction, deep caries, and periapical lesion are handled as healthy versus severe.
- This supports clinical triage by urgency.
- Next, I will cover augmentation and post-processing improvements.

---

## Slide 13 - Improvements D and E: CLAHE + Quadrant Penalty

- For augmentation, I use CLAHE in Phase 3 with probability 0.5.
- It is applied on the L channel in LAB space, with clip limit 2.0 and grid 8x8.
- This helps normalize contrast variability across X-ray machines.
- For post-processing, I use quadrant-consistency penalty inside the Hungarian assignment cost.
- This specifically reduces symmetric left-right swaps.
- With alpha 0.5, quadrant information acts as a tie-breaker.
- Midline swap rate drops from 8.2 percent to 4.8 percent, a 41 percent relative reduction.
- Next, I will walk through the training protocol.

---

## Slide 14 - Three-Phase Curriculum Training Protocol

- Phase 1: 100 epochs, SGD, lr 0.01, full labelled detection pretraining.
- Best Phase 1 mAP50 is 0.5128 at epoch 55, and training takes around 61.5 minutes on T4.
- Phase 2: 53 epochs, SGD, lr 0.002, fine-tuning with pseudo-labelled expansion.
- Best Phase 2 mAP50 is 0.5373 at epoch 5, then overfitting appears, so I keep the early best checkpoint.
- Phase 2b: 50 epochs on Part 3 attribute tuning with frozen backbone; best BCE is 2.3907.
- Phase 3: 50 epochs, AdamW, lr 5e-4, train Swin plus cross-attention plus severity head with frozen detector core.
- Hybrid loss reduces from 1.93 to 1.45, around 25.1 percent improvement.
- Next, I will show the live inference pipeline.

---

## Slide 15 - Live Demo

- I run prediction using: python main.py --mode predict --image <path>
- Input can be any panoramic PNG or JPEG.
- Output includes a structured JSON report and an annotated image.
- Example: FDI 16 is detected as Upper Right First Molar with mild caries probabilities.
- If time allows, I run one sample live.
- Next, I will briefly mention code and reproducibility.

---

## Slide 16 - Code Base and Pipeline Links

- The full source is available on GitHub in the archon-working branch.
- The pipeline is also reproducible on Google Colab.
- I can share links on request.
- Now I will move to quantitative results.

---

## Slide 17 - Results: Phase 1 Training Curves

- Best Phase 1 mAP50 is 0.5128 at epoch 55.
- mAP50-95 is 0.3415.
- Precision is 0.5292 and recall is 0.5171 at best epoch.
- Peak F1 is 0.45 at confidence 0.377.
- On this dataset scale and 32-class FDI task, this is competitive.
- This phase corresponds to the baseline-equivalent setup.
- Next, I will show Phase 2 behavior.

---

## Slide 18 - Results: Phase 2 Fine-tuning

- In Phase 2, mAP50 peaks early at 0.5373 around epoch 5.
- After that, validation loss rises steadily, indicating overfitting.
- The cause is smaller disease-rich data volume in Part 3.
- So I retain early best checkpoint and continue with frozen-backbone strategy.
- In Phase 2b, best attribute BCE is 2.3907 at epoch 48.
- The attribute weights are tuned for imbalance.
- Next, I will compare PR and F1 curves.

---

## Slide 19 - Results: PR and F1 Curves

- Phase 1 PR mAP50 is 0.5128.
- Full ARCHON eval PR mAP50 is 0.4991.
- Phase 1 F1 peak is 0.45 at confidence 0.377.
- ARCHON F1 peak is 0.38 at confidence 0.072.
- The slight mAP difference comes from different split and full post-processing pipeline.
- Clinically, threshold-specific performance is more relevant, especially near conf 0.15.
- Next, I will show confusion matrices.

---

## Slide 20 - Results: Confusion Matrices

- These are normalized confusion matrices for 32 FDI classes.
- Lower molars show strong diagonal performance.
- Background confusions are expected on a small and imbalanced validation set.
- Importantly, detection quality remains stable across phases.
- Next, I will show qualitative examples.

---

## Slide 21 - Visual Demo: Input vs Inferred vs Combined

- Here I show raw input, model overlay, and combined comparison view.
- For train_23 and train_130, detections and disease flags are fully correct in these best-case examples.
- Clinical thresholds here are conf 0.15 and attr 0.08.
- These are best demos, but they show the model can achieve perfect per-image output.
- Next, I will show full validation batches.

---

## Slide 22 - Validation Batch 0 and 1

- These are full validation batch visuals from training, not cherry-picked.
- We can see generally correct box placement.
- Remaining errors are mostly symmetric FDI naming confusion.
- Next is batch 2.

---

## Slide 23 - Validation Batch 2

- Batch 2 shows the same pattern: good localization with occasional left-right FDI mixups.
- Next, I will show representative JSON output.

---

## Slide 24 - Representative Inference Output (JSON)

- The JSON gives total_teeth and diseased_teeth summary.
- It also groups detections by quadrant.
- Each tooth contains FDI, readable name, confidence, bbox, disease flags, and severity details.
- Example: FDI 16 is mild caries with healthy/mild/severe probabilities.
- FDI 37 has high confidence, consistent with strong lower molar performance.
- This output is structured for downstream EMR integration.
- Next, I will discuss severity convergence.

---

## Slide 25 - Severity Grading and Phase 3 Convergence

- Phase 3 loss converges smoothly and remains stable.
- Attribute head BCE reaches best around epoch 39.
- Severe imbalance exists for periapical lesion positives, so I use higher positive weighting.
- Severity distribution is naturally skewed toward healthy, matching clinical reality.
- The head design is stable and learns meaningful severity signals.
- Next, I compare ARCHON with existing methods.

---

## Slide 26 - Comparative Study

- Classical threshold methods have no learning or severity understanding.
- Mask R-CNN is accurate but too slow for real-time workflow.
- YOLOv5 is faster but less accurate.
- YOLOrtho is the strongest baseline but lacks severity and full-arch global reasoning.
- ARCHON adds severity and global context while retaining real-time practical speed around 26 FPS.
- Reported mAP is slightly lower partly because this run uses a smaller training subset.
- Next, I will show threshold comparison on 700 GT-validated images.

---

## Slide 27 - Threshold Comparison (700 Images)

- I tested six threshold combinations on 700 images with ground truth.
- The best overall F1 score comes from conf 0.05 and attr 0.10.
- The best clinical trade-off is conf 0.15 and attr 0.08.
- Very high thresholds reduce false positives but miss many teeth.
- For clinical usability, I choose conf 0.15 and attr 0.08 because precision is high and rework is lower.
- Next, I will show the detailed numbers at this clinical threshold.

---

## Slide 28 - Best Combination Accuracy Details

- Tooth detection at conf 0.15 and attr 0.08: recall 75.03 percent, precision 92.38 percent, F1 82.81 percent.
- Counts are TP 2629, FP 217, FN 802 out of 3504 GT teeth.
- Disease detection: recall 69.24 percent, precision 53.34 percent, F1 60.26 percent.
- Counts are TP 2422, FP 2119, FN 1076 out of 3498 GT diseases.
- The selected threshold sharply reduces tooth false positives versus aggressive settings.
- Next, I will discuss computational cost.

---

## Slide 29 - Computational Performance Analysis

- Compared to YOLOrtho, ARCHON increases parameters from 68.5M to 79.5M.
- FLOPs increase from 187 to 215 GFLOPs.
- GPU memory rises from 7.8 GB to 9.4 GB.
- Inference latency increases from 31 ms to 38 ms.
- Total training time is about 362 minutes on one NVIDIA T4.
- In practice, the added 7 ms per image is negligible in dental workflow.
- ARCHON remains real-time capable.
- Next, I will summarize the improvements.

---

## Slide 30 - Improvements in Numbers

- Phase 1 mAP50: 0.5128.
- Phase 2 best mAP50: 0.5373, a 2.45-point gain.
- Hybrid loss reduction: 25.1 percent.
- Midline swaps reduced from 8.2 percent to 4.8 percent.
- Clinical tooth precision: 92.38 percent.
- Best demo cases achieve exact matching.
- All these values are from logged runs and validation reports.
- Next, I will conclude.

---

## Slide 31 - Conclusion

- I will close with six contributions.
- One, global context encoder reduces midline swaps.
- Two, multiscale fusion improves disease-context learning.
- Three, hybrid multi-task head enables clinically useful severity output.
- Four, CLAHE improves robustness to varying image exposure.
- Five, quadrant consistency further reduces symmetric confusion.
- Six, the curriculum plus freezing strategy protects detection performance.
- For future work, I plan full-dataset scaling, stronger severity labels, and multi-center validation.
- Next, I will be transparent about current limitations.

---

## Slide 32 - Limitations and Recovery Plan

- This run uses a subset, not full DENTEX, so scaling is the immediate next step.
- Phase 2 shows overfitting, so I will strengthen early stopping and cross-validation.
- Severity labels are still proxy-based and need radiologist verification.
- Rare lesion classes remain imbalanced, so I will improve reweighting and hard-sample mining.
- Prospective clinical validation is pending and planned as multi-center evaluation.
- These are known engineering limitations with clear recovery paths.
- Thank you, and I will now close.

---

## Slide 33 - Thank You

- Thank you for your time and attention.
- I am happy to take your questions.

---

## Slide 34 - Any Questions?

- I am ready to answer on training time, threshold logic, pseudo-label quality, mAP differences, and deployment workflow.

---

## General Delivery Tips (Same Tone)

1. Speak as if explaining your own engineering decisions, not reading bullet points.
2. Keep transitions simple: "Next I will show...", "Now I will explain...", "The key point is...".
3. Emphasize numbers only where they support a decision.
4. Be honest on limitations and immediately mention your recovery plan.
5. For difficult questions, answer in three steps: what happened, why, and what I will do next.
