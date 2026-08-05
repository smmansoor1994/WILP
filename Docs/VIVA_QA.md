# ARCHON Thesis — Anticipated Viva Questions, Answers & Explanations

This document covers likely questions from the panel with detailed answers you can prepare.

---

## Category 1: Architecture & Design Choices

### Q1. Why did you choose YOLOv8x as the backbone instead of other architectures like ResNet, EfficientDet, or DETR?

**Answer:**
YOLOv8x was chosen for three reasons:
1. **32-class discriminative capacity**: The largest YOLO variant (68.5M params) provides sufficient feature depth for distinguishing 32 FDI tooth classes that are morphologically similar.
2. **Real-time inference**: At ~31ms/image on T4, it meets clinical deployment requirements (≥25 FPS).
3. **Ultralytics ecosystem**: Built-in COCO metrics, checkpoint management, and training utilities reduced engineering overhead.

DETR was considered but its O(N²) attention on 800+ tokens would be prohibitively expensive. EfficientDet's BiFPN is lighter but lacks the stride-8 fine-grained detection needed for incisors.

**Deeper explanation:** ARCHON inherits from YOLOrtho (Mei et al., 2023) which validated YOLOv8 for dental panoramic detection. Changing the backbone would invalidate the baseline comparison and require re-establishing all detection benchmarks.

---

### Q2. Why use Swin Transformer only on P5 and not on all feature levels?

**Answer:**
- P5 has 800 tokens (20×40) ≈ 1 token per tooth — ideal granularity for inter-tooth relationships.
- Applying Swin to P3 (160×80 = 12,800 tokens) would be computationally prohibitive.
- P5's coarse semantic level is where quadrant/arch-level decisions are made.
- Local texture features at P3/P4 don't benefit from global context — they need fine detail.

The global context is then propagated to P3/P4 via cross-attention fusion, so all scales still benefit from Swin's output.

---

### Q3. Explain the residual connection in MultiScaleFusion. Why is it necessary?

**Answer:**
The residual connection ensures:
```
output = CNN_features + CrossAttention(CNN_features, Swin_context)
```

If the cross-attention module produces no useful signal (e.g., early training when Swin weights are random), the output degrades gracefully to the original CNN features. This provides a mathematical guarantee that ARCHON can never perform worse than the CNN-only baseline. Without it, random Swin outputs in early training could destroy good detection features.

---

### Q4. Why did you choose window size 4 for Swin instead of 7 (the default)?

**Answer:**
P5 is 20×40 tokens. Window size 7 would create irregular tiling (20 is not divisible by 7), requiring padding and wasting computation. Window size 4 tiles perfectly into 5×10 = 50 windows. Additionally, each window covers ~4 adjacent teeth — exactly the spatial scale needed for quadrant-level relationships.

---

### Q5. How does CoordConv help with FDI numbering specifically?

**Answer:**
FDI numbering is purely position-dependent: tooth 16 and tooth 26 are mirror images across the midline — identical in texture, shape, and size. A standard CNN cannot distinguish them because convolutions are translation-equivariant (same filter response regardless of position). CoordConv prepends normalised (x, y) ∈ [-1, +1] coordinate channels, allowing the network to learn "this tooth is on the left side → quadrant 2" vs "right side → quadrant 1."

---

### Q6. Why a hybrid multi-task head instead of separate models for detection and disease classification?

**Answer:**
1. **Efficiency**: Single forward pass vs two sequential models — halves latency.
2. **Shared features**: Detection and disease share backbone representations; joint training improves both.
3. **Clinical workflow**: A single report with detection + severity is more actionable than running two separate tools.
4. **Alignment**: Disease is localised to a specific tooth — the detection head provides the exact ROI.

The downside (potential negative transfer) is mitigated by the frozen-backbone strategy in Phase 3.

---

## Category 2: Training Strategy

### Q7. Why curriculum learning with 4 phases instead of training everything end-to-end?

**Answer:**
DENTEX has hierarchical annotations — not all images have disease labels. End-to-end training would either:
- Waste data (only use Part 3 with full labels) — losing 800+ detection-only images.
- Create label noise (assign "no disease" to images where disease was simply not annotated).

Curriculum learning respects this hierarchy: Phase 1 uses ALL data for detection, then Phases 2-3 progressively add disease/severity using appropriate subsets.

---

### Q8. Phase 2 overfits after epoch 5. How did you handle this, and what would you do differently?

**Answer:**
**How we handled it:** Saved the early best checkpoint (epoch 5, mAP50=0.5373) and used that as the base for subsequent phases. The frozen-backbone design in Phase 2b/3 ensures this detection quality is preserved.

**What I would do differently:**
- Implement k-fold cross-validation (5-fold) to use all Part 3 data more effectively.
- Use stronger regularisation: mixup augmentation, larger dropout in attribute heads.
- Reduce learning rate warmup and use cosine decay from epoch 1.
- Consider fewer epochs with larger batch accumulation to smooth gradients.

---

### Q9. Why freeze the backbone in Phase 3? Doesn't this limit learning?

**Answer:**
When Phase 3 begins, the Swin Transformer and CrossAttention modules have random weights. Their gradients are noisy and large. If these gradients backpropagate through the backbone, they corrupt the well-trained detection features — empirically causing 4-6pp mAP50 drop in early epochs.

Freezing the backbone guarantees:
1. Detection performance (mAP50=0.5373) is preserved exactly.
2. Only the new modules (Swin, CrossAttn, SeverityHead) learn.
3. Gradient magnitude is controlled — no catastrophic forgetting.

**Limitation acknowledged:** This means the backbone cannot adapt to severity-specific features. In future work, with a larger dataset, unfreezing with very low LR (1e-5) could allow end-to-end fine-tuning after Phase 3 converges.

---

### Q10. How are pseudo-labels generated and what quality control is applied?

**Answer:**
1. Phase 1 model runs inference on the ~300 unlabelled images.
2. Only detections with confidence > 0.5 are retained as pseudo-labels.
3. Non-maximum suppression (IoU=0.45) removes duplicates.
4. Labels are saved in YOLO format (class x_center y_center width height).

Quality control: The confidence threshold (0.5) ensures only high-confidence predictions become training data. This is a conservative threshold — higher than the evaluation threshold — to avoid propagating errors.

**Known limitation:** No human verification was done. In production, a radiologist-in-the-loop would validate pseudo-labels before Phase 2 training.

---

### Q11. What is the learning rate schedule and why those specific values?

**Answer:**
- **Phase 1:** SGD, lr=0.01 with linear warmup (3 epochs) → cosine decay. Standard YOLO training recipe.
- **Phase 2:** SGD, lr=0.002 (5× lower). Lower LR because we're fine-tuning a pre-trained model — large LR would destroy Phase 1 features.
- **Phase 3:** AdamW, lr=5e-4 with CosineAnnealingLR. AdamW chosen because Transformer modules (Swin) benefit from adaptive per-parameter learning rates. SGD struggles with attention weight initialisation.

---

## Category 3: Dataset & Evaluation

### Q12. Why is your mAP50 (0.4991) lower than published YOLOrtho results (~0.61)?

**Answer:**
Three key differences:
1. **Dataset size**: ARCHON trained on ~2000 images (subset). YOLOrtho published results use full DENTEX (4000+ images). More data → higher mAP.
2. **Task complexity**: ARCHON evaluates full pipeline (detection + disease + severity), while YOLOrtho evaluates detection only.
3. **Evaluation split**: Our 250-image test set may have different difficulty distribution.

The fair comparison is Phase 1 (same architecture, same data) = 0.5128 mAP50 vs ARCHON full eval = 0.4991. The small drop (1.37pp) comes from post-processing overhead, not architecture degradation.

---

### Q13. How did you determine the clinical threshold (conf=0.15, attr=0.08)?

**Answer:**
We ran a systematic grid search over 6 threshold combinations on 700 validation images with full ground truth:
- Evaluated all combinations of conf ∈ {0.05, 0.08, 0.15, 0.20, 0.25} × attr ∈ {0.05, 0.08, 0.10, 0.12, 0.20}.
- Ranked by overall score (weighted average of Tooth F1 and Disease F1).

conf=0.15, attr=0.08 was selected because:
- Tooth precision = 92.38% — critical for clinical trust (low false alarm rate).
- Tooth FP drops from 872 to 217 vs lower threshold (75% reduction).
- Disease recall = 69.24% — still catches most pathology.
- Balanced trade-off suitable for radiology triage workflow.

---

### Q14. Your disease precision is only 53.34%. Isn't that too low for clinical use?

**Answer:**
In radiology AI, **sensitivity (recall) is prioritised over precision** for disease screening:
- A false positive means the radiologist reviews a healthy area — minor time cost.
- A false negative means missing a disease — potential patient harm.

53.34% precision means ~half of disease flags are false positives. This is acceptable in a **triage/screening** setting where:
1. The AI highlights suspicious regions.
2. The radiologist makes the final diagnosis.
3. It's used as a "second reader" — not autonomous diagnosis.

For autonomous diagnosis (no human review), we would need precision >80%, requiring more training data and annotation refinement.

---

### Q15. Why use IoU threshold 0.45 for validation instead of the standard 0.50?

**Answer:**
Dental bounding boxes on panoramic X-rays are inherently imprecise because:
- Tooth boundaries overlap (contact points).
- Root apices are poorly defined in periapical regions.
- Annotation variability between dentists is high (inter-annotator IoU ~0.6-0.7).

IoU=0.45 is slightly more lenient to account for this annotation noise. The DENTEX challenge also uses relaxed IoU thresholds for the same reason.

---

### Q16. How does the DENTEX dataset compare to real clinical data?

**Answer:**
**Similarities:**
- Standard panoramic X-rays from clinical centres.
- Realistic disease distribution (mostly healthy, some pathology).
- Professional annotation by dental radiologists.

**Differences from real-world:**
- DENTEX is cross-sectional (single timepoint) — no longitudinal tracking.
- Limited demographic diversity (single/few centres).
- Disease labels are binary (present/absent) — no ICDAS severity scale.
- ~1200 labelled images is small compared to clinical archives (millions).

This is why prospective multi-centre validation is listed as future work.

---

## Category 4: Technical Deep-Dives

### Q17. Explain the Hungarian algorithm in the context of FDI assignment.

**Answer:**
After YOLO detects bounding boxes, each box has a 32-class probability vector P_cls. We need to assign each detection to a unique FDI number (one-to-one mapping).

**Standard approach:**
- Create a cost matrix: cost[i,j] = -log(P_cls[detection_j, class_i]).
- Run Hungarian algorithm to find minimum-cost one-to-one assignment.

**ARCHON enhancement:**
- Add quadrant penalty: cost[i,j] += α × (1 - P_quad[j, quadrant(i)]).
- This penalises assigning a tooth to a class whose quadrant doesn't match the predicted quadrant.
- Result: symmetric teeth that are spatially distinct get correctly assigned to their respective quadrants.

---

### Q18. What is DFL (Distribution Focal Loss) and why is it important?

**Answer:**
Standard bounding box regression predicts 4 values (x, y, w, h) as point estimates. DFL instead predicts a discrete probability distribution over possible coordinate values.

**Why it helps:**
- Ambiguous boundaries (overlapping teeth) produce a bimodal distribution instead of forcing a single estimate.
- The loss focuses on the most uncertain boundaries (focal property).
- Produces tighter boxes on clear boundaries and appropriately uncertain boxes on ambiguous ones.

This is particularly important in dental imaging where tooth contact points create ambiguous boundaries.

---

### Q19. How does the severity head architecture work internally?

**Answer:**
```
Input: pooled features from hybrid head (per-detection)
  → Linear(feature_dim, 128)
  → ReLU
  → Dropout(0.3)
  → Linear(128, 3) per disease  [healthy, mild, severe]
  → Softmax → probability distribution
```

Four parallel branches (one per disease). Each outputs a 3-class distribution. Loss is CrossEntropyLoss with class weights to handle imbalance.

The severity labels are derived from binary DENTEX flags:
- Caries: has_caries=1 → mild; has_deepcaries=1 → severe.
- Impaction/Lesion: binary → directly severe if present.

---

### Q20. Explain CLAHE in detail. Why clip limit 2.0 and tile grid 8×8?

**Answer:**
CLAHE = Contrast-Limited Adaptive Histogram Equalisation.

**Process:**
1. Divide image into 8×8 tiles (grid).
2. Compute histogram per tile.
3. Clip histogram at limit 2.0 (redistribute excess equally).
4. Equalise each tile's histogram independently.
5. Bilinear interpolate at tile boundaries for smooth result.

**Why clip limit 2.0:** Higher values (e.g., 4.0) over-amplify noise in uniform regions (bone areas in X-rays). Lower values (e.g., 1.0) barely enhance contrast. 2.0 is empirically established for medical imaging.

**Why 8×8 grid:** Standard medical imaging choice. Smaller grids (16×16) adapt too locally and create artificial edges. Larger grids (4×4) behave like global histogram equalisation.

**Why L-channel only:** X-rays are grayscale stored as 3-channel BGR. Applying CLAHE to B, G, R independently creates colour shifts. L-channel in LAB space isolates luminance — modifying only brightness without affecting (already non-existent) colour.

---

## Category 5: Results Interpretation

### Q21. The confusion matrix shows high background rate. Is this a problem?

**Answer:**
The high "background" column in the confusion matrix means many ground-truth teeth are predicted as background (i.e., missed). This is expected because:
1. **32-class imbalance:** With 32 classes on a small val set, many classes have <5 samples → low confidence → predicted as background.
2. **Confidence threshold:** At default YOLO threshold (0.25), many low-confidence detections are suppressed.
3. **Small validation set:** 50 val images × ~5 teeth average = ~250 teeth across 32 classes ≈ 8 per class.

**It's not a fundamental problem** — it's an artifact of evaluation methodology. At clinical threshold (0.15), most of these become true detections.

---

### Q22. Why does Phase 2 overfit so quickly (epoch 5)?

**Answer:**
Root causes:
1. **Small dataset:** Part 3 has only ~400 images with disease labels — with data augmentation giving effective ~600-800 samples.
2. **Model capacity mismatch:** 79.5M parameter model on 400 images → ratio is 200K params/image — far above the typical 100:1 rule.
3. **Shifted distribution:** Phase 2 data includes pseudo-labels which may have distribution shift from real labels.
4. **No dropout in backbone:** YOLOv8 backbone has no dropout layers — regularisation relies solely on augmentation.

**Mitigation applied:** Early stopping (save best at epoch 5), then frozen backbone for subsequent phases.

---

### Q23. Explain the difference between mAP50 and mAP50-95.

**Answer:**
- **mAP50:** Average Precision computed at IoU threshold = 0.50. A detection is "correct" if IoU with GT ≥ 0.50.
- **mAP50-95:** Average of AP computed at IoU thresholds [0.50, 0.55, 0.60, ..., 0.95]. Much stricter — requires very tight bounding boxes at high thresholds.

ARCHON achieves:
- mAP50 = 0.5128 (Phase 1) — reasonable for 32-class dental.
- mAP50-95 = 0.3415 — lower because dental tooth boundaries are inherently imprecise (overlapping teeth, fuzzy roots).

The mAP50 metric is more relevant for clinical use because:
- Exact box tightness matters less than correct tooth identification.
- IoU>0.5 is sufficient to localise which tooth is being referenced.

---

### Q24. What does the F1 peak at conf=0.377 (Phase 1) vs conf=0.072 (ARCHON eval) tell us?

**Answer:**
- **Phase 1 (conf=0.377):** The model is well-calibrated for detection — optimal confidence threshold is moderate, meaning the model makes distinct high/low confidence predictions.
- **ARCHON eval (conf=0.072):** The optimal threshold is very low — the model is making many borderline predictions. This suggests the full pipeline (with severity/disease heads) slightly reduces detection confidence.

**Clinical implication:** At conf=0.15 (clinical threshold), we're above the ARCHON eval F1 peak — deliberately sacrificing some recall for much higher precision. This is the correct trade-off for clinical deployment.

---

## Category 6: Clinical Relevance

### Q25. How would ARCHON be deployed in a real clinical setting?

**Answer:**
Deployment workflow:
1. **Integration:** ARCHON runs as a DICOM listener or PACS plugin.
2. **Trigger:** When a new panoramic X-ray arrives, inference runs automatically (~38ms).
3. **Output:** Structured JSON report + annotated image displayed alongside original.
4. **Role:** Second reader — highlights potential findings for radiologist review.
5. **Feedback loop:** Radiologist confirms/rejects findings → retraining data.

**NOT autonomous diagnosis** — regulatory frameworks (FDA 510(k), CE MDR) require human-in-the-loop for dental AI as of 2026.

---

### Q26. What is the clinical significance of 3-level severity over binary detection?

**Answer:**
Binary (disease present/absent) tells the dentist WHAT but not HOW URGENT:
- **Mild caries** → watch and wait, preventive fluoride, 6-month review.
- **Severe caries** → immediate intervention, possible root canal.

Without severity, every detection has equal priority. With 3-level grading:
- Triage becomes automated — severe cases flagged immediately.
- Treatment planning is more accurate — conservative vs aggressive approach.
- Patient communication improves — "mild" is less alarming than "disease detected."

This maps to clinical ICDAS scale (International Caries Detection and Assessment System) which uses 0-6 severity levels.

---

### Q27. How does the quadrant-consistency penalty relate to real dental anatomy?

**Answer:**
The human dentition is divided into 4 quadrants:
- Q1: Upper Right (teeth 11-18)
- Q2: Upper Left (teeth 21-28)
- Q3: Lower Left (teeth 31-38)
- Q4: Lower Right (teeth 41-48)

Teeth across the midline (e.g., 16 and 26) are mirror images — nearly identical in shape. A standard classifier often confuses them. The quadrant penalty leverages the spatial position of the detection: if a box is on the right side of the image, assigning it to Q2 (left) is penalised.

This is anatomically grounded — you cannot have a Q1 tooth on the left side of a correctly-oriented panoramic X-ray.

---

### Q28. What diseases does ARCHON detect and why these four?

**Answer:**
ARCHON detects:
1. **Caries** — most common dental disease (60-90% prevalence globally).
2. **Deep caries** — advanced caries reaching pulp — urgent intervention needed.
3. **Impaction** — tooth unable to erupt normally — surgical planning required.
4. **Periapical lesion** — infection at tooth root apex — indicates endodontic pathology.

These four were chosen because:
- They are the diseases annotated in DENTEX 2023 dataset.
- They represent the most common radiographically-visible pathologies.
- They cover different urgency levels (screening + urgent).
- They are reliably detectable on panoramic X-rays (vs. periodontal disease which needs periapical views).

---

## Category 7: Limitations & Future Work

### Q29. What are the biggest limitations of your work?

**Answer (be honest and structured):**
1. **Dataset scale:** Trained on ~2000 images vs full DENTEX (4000+). Limits mAP ceiling.
2. **Severity labels are proxy:** Derived from binary flags, not radiologist-verified ICDAS grades.
3. **No prospective validation:** Tested on DENTEX only — no real clinical site evaluation.
4. **Phase 2 overfitting:** Small disease-annotated subset causes quick overfitting.
5. **Class imbalance:** Periapical lesion has only 3 positive samples in validation — statistics are unreliable.
6. **Single modality:** Panoramic X-rays only — no CBCT, periapical, or bitewing integration.

**Each has a concrete recovery path** (as shown in the limitations slide).

---

### Q30. If you had 6 more months, what would you prioritise?

**Answer:**
Priority order:
1. **Full DENTEX training** — immediate 3-5pp mAP50 improvement expected.
2. **K-fold cross-validation** — address overfitting, get reliable severity statistics.
3. **Radiologist severity annotation** — 200 images with ICDAS grades to validate severity head.
4. **Multi-centre evaluation** — 2-3 clinical sites, different X-ray machines, ~500 new images.
5. **Attention visualisation** — Grad-CAM on Swin output to show clinical interpretability.

---

### Q31. How would you address the class imbalance problem?

**Answer:**
Current approach: pos_weight in BCE loss (e.g., 20.0 for lesion). This is a basic solution.

Better approaches (future work):
1. **Focal Loss:** Down-weight easy negatives, focus on hard examples.
2. **SMOTE for features:** Generate synthetic positive features in embedding space.
3. **Hard negative mining:** Focus training on confusing cases.
4. **External data:** Collect additional lesion-positive images from clinical archives.
5. **Few-shot learning:** Meta-learning approach for rare disease classes.
6. **Class-balanced sampling:** Oversample mini-batches to include proportional positives.

---

### Q32. How would you validate ARCHON for regulatory approval (FDA/CE)?

**Answer:**
Regulatory pathway (FDA 510(k) for Computer-Aided Detection):
1. **Predicate device:** Cite existing cleared dental AI (e.g., Overjet, Pearl) as substantial equivalence.
2. **Clinical validation study:**
   - Prospective, multi-site (≥3 centres).
   - ≥500 patients, diverse demographics.
   - Ground truth by consensus of 3 board-certified radiologists.
   - Primary endpoints: sensitivity ≥80%, specificity ≥70% for disease detection.
3. **Standalone vs reader study:** Show AI+radiologist > radiologist alone (reader study design).
4. **Cybersecurity & data privacy:** HIPAA compliance, DICOM de-identification pipeline.
5. **Post-market surveillance:** Monitoring for dataset drift over time.

---

## Category 8: Comparison & Literature

### Q33. How does ARCHON compare to the DENTEX challenge winners?

**Answer:**
DENTEX 2023 challenge winning solutions typically achieved:
- Disease detection mAP ~0.45-0.55 (comparable approaches).
- Used ensemble methods (3-5 models) — ARCHON is a single model.
- Did NOT address severity grading or full FDI enumeration.
- Trained on full DENTEX dataset (vs ARCHON's subset).

**ARCHON's unique value:** It's not trying to beat DENTEX leaderboard on a single metric — it provides a complete clinical solution (enumeration + disease + severity) that no challenge winner addressed.

---

### Q34. Why not use a Vision Transformer (ViT) instead of CNN + Swin hybrid?

**Answer:**
1. **Data efficiency:** ViTs need large datasets (ImageNet-21k pre-training or 100k+ images). With ~2000 dental images, CNN inductive biases (locality, translation equivariance) are essential.
2. **Multi-scale detection:** YOLO's FPN naturally produces multi-scale features. Pure ViT requires complex adaptations (Swin-V2, ViT-Det) for detection.
3. **Proven baseline:** YOLOv8 is validated for dental detection. Replacing with ViT would require re-establishing all baselines.
4. **Hybrid approach:** ARCHON gets the best of both — CNN for efficient local features + Swin for global context where needed (P5 only).

---

### Q35. What is the novelty of your work compared to YOLOrtho?

**Answer:**
YOLOrtho (Mei et al.) provides:
- YOLOv8 + CoordConv + FPN for detection.
- Hungarian algorithm for FDI assignment.
- No disease detection, no severity, no global context.

ARCHON adds FIVE novel components:
1. **GlobalContextEncoder** (Swin on P5) — full-arch awareness.
2. **MultiScaleFusion** (Cross-Attention) — inject global context at all scales.
3. **HybridMultiTaskHead** — 3-level severity × 4 diseases.
4. **CLAHE augmentation** — exposure robustness.
5. **Quadrant-Consistency Penalty** — symmetric tooth disambiguation.

Plus the **3-phase curriculum training protocol** which enables all components to train without regression.

---

## Category 9: Implementation Details

### Q36. How long does training take and what hardware did you use?

**Answer:**
- **Hardware:** Single NVIDIA T4 GPU (16GB VRAM), Google Colab / local workstation.
- **Phase 1:** 100 epochs, ~61.5 minutes.
- **Phase 2:** 53 epochs, ~63.7 minutes.
- **Phase 2b:** 50 epochs, ~97.6 minutes.
- **Phase 3:** 50 epochs, ~98.9 minutes.
- **Total:** ~362 minutes (~6 hours) end-to-end.

Batch size: 4 (limited by 16GB VRAM at 1280×640 resolution). Larger batch would smooth training but requires multi-GPU.

---

### Q37. How do you handle different X-ray resolutions and orientations?

**Answer:**
- **Resolution:** All images resized to 1280×640 with letterbox padding (preserves aspect ratio).
- **Orientation:** DENTEX provides consistently-oriented OPGs. In deployment, DICOM metadata provides orientation tags.
- **Augmentation:** Random horizontal flip is NOT used because it would swap quadrants (Q1↔Q2, Q3↔Q4) and invalidate FDI labels.
- **Other augmentations:** Mosaic, mixup, random scale (0.5-1.5), HSV shift, CLAHE.

---

### Q38. What framework and libraries does your implementation use?

**Answer:**
- **Core:** Ultralytics YOLOv8 (v8.0+), PyTorch 2.x.
- **Custom modules:** Swin Transformer (adapted from Microsoft's implementation), custom heads.
- **Training:** SGD/AdamW optimisers from PyTorch.
- **Evaluation:** COCO metrics via Ultralytics + custom validation script.
- **Deployment:** python-pptx for reporting, JSON for structured output.
- **Development environment:** Python 3.11+, CUDA 11.8+, 16GB GPU.

---

## Category 10: Statistical & Methodological Rigour

### Q39. With only 50 validation images, how reliable are your metrics?

**Answer:**
**Honest answer:** 50 validation images is statistically weak for 32-class evaluation. This gives ~1-2 images per class on average — insufficient for confident per-class metrics.

**Mitigations:**
1. The 700-image GT validation report provides much stronger evidence (3504 GT teeth, 3498 GT diseases).
2. We report aggregate metrics (overall mAP, F1) rather than per-class.
3. Confusion matrix analysis focuses on groups (molars, premolars) rather than individual classes.
4. The threshold analysis uses the full 700-image set — statistically meaningful.

**Future improvement:** Use k-fold cross-validation or a dedicated 200+ image test set.

---

### Q40. How would you compute confidence intervals for your reported metrics?

**Answer:**
For binary metrics (precision, recall, F1) on the 700-image validation set:
- Use Wilson score interval or Clopper-Pearson exact interval.
- Example: Tooth precision = 92.38% with n=2846 predictions → 95% CI: [91.3%, 93.3%].
- Disease recall = 69.24% with n=3498 GT → 95% CI: [67.7%, 70.8%].

For mAP: Bootstrap resampling (1000 iterations) over the test set would give confidence bounds.

**Why not reported:** Standard practice in YOLO-based papers is to report point estimates. Adding CIs would strengthen the thesis but was not done in this version.

---

## Quick Reference — Numbers to Memorise

| Metric | Value | Context |
|--------|-------|---------|
| Phase 1 mAP50 | 0.5128 | Best epoch 55/100 |
| Phase 2 mAP50 | 0.5373 | Best epoch 5/53 |
| Full eval mAP50 | 0.4991 | archon_best.pt |
| Clinical Tooth P/R/F1 | 92.38 / 75.03 / 82.81 | conf=0.15, attr=0.08 |
| Clinical Disease P/R/F1 | 53.34 / 69.24 / 60.26 | conf=0.15, attr=0.08 |
| Midline swap reduction | 8.2% → 4.8% | 41% relative |
| Hybrid loss drop | 25.1% | 1.93 → 1.45 |
| Parameters | ~79.5M | vs 68.5M baseline |
| Inference latency | 38ms | on T4 GPU |
| Total training time | ~362 min | 4 phases combined |
| Swin complexity saving | 50× | vs full self-attention |
| Dataset: train/val/test | 1500/50/250 | images |
| GT validation scope | 700 images | 3504 teeth, 3498 diseases |

---

## Tips for Handling Difficult Questions

1. **If you don't know:** "That's an excellent question. Based on my analysis, I would hypothesise [X], though I haven't empirically verified this. It would be a good direction for future work."

2. **If asked about a limitation:** Acknowledge it directly, then pivot to the recovery plan. Never be defensive.

3. **If asked to compare with a paper you haven't read:** "I'm not familiar with that specific paper, but based on the approach you describe, the key difference would be [architectural/methodological distinction]."

4. **If asked about clinical deployment concerns:** Always mention: regulatory requirements, human-in-the-loop, and post-market surveillance.

5. **If asked "what would you do differently?":** Focus on data (more data, better labels, cross-validation) rather than architecture — shows practical maturity.
