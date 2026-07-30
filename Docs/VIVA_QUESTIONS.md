# ARCHON — Viva Question Bank (60 Questions)

> 3 Levels × 20 Questions each — Scenario-based, practical, examiner-style.
> Practice answering these out loud before your viva.

---

## Level 1: EASY (Fundamentals & Definitions)

These test whether you understand the basic building blocks of your project.

---

### Q1. What does the acronym ARCHON stand for and what is the project's goal?

**Expected Answer**: ARCHON = Arch-Contextualized Hierarchical Orthodontic Network. The goal is to simultaneously detect teeth, assign FDI numbers (1–32), and diagnose four dental diseases (impaction, caries, deep caries, periapical lesion) from panoramic dental X-rays.

---

### Q2. What is the FDI numbering system? Give an example.

**Expected Answer**: FDI (Fédération Dentaire Internationale) assigns a 2-digit number to each tooth. First digit = quadrant (1=Upper Right, 2=Upper Left, 3=Lower Left, 4=Lower Right), second digit = position (1=central incisor to 8=wisdom tooth). Example: FDI 36 = Lower Left First Molar.

---

### Q3. How many classes does your model detect and why?

**Expected Answer**: 32 classes — one for each possible adult tooth (4 quadrants × 8 positions). We use class-level detection rather than quadrant+position regression because each tooth needs a unique identity for the linear sum assignment post-processing.

---

### Q4. What is YOLOv8 and why is it called "You Only Look Once"?

**Expected Answer**: YOLO is a one-stage object detector that processes the entire image in a single forward pass, outputting all bounding boxes and class predictions simultaneously — unlike two-stage detectors (R-CNN) that first propose regions then classify them. "Look Once" = one pass through the network.

---

### Q5. What are the three main parts of any object detector?

**Expected Answer**: Backbone (extracts features from image), Neck (merges features from different scales via FPN), Head (produces final predictions — bounding boxes + classes).

---

### Q6. What is a bounding box and what format do we use?

**Expected Answer**: A rectangle around a detected object. We use YOLO format: `cx cy w h` (center-x, center-y, width, height), all normalized to [0, 1] relative to image dimensions.

---

### Q7. What are the four diseases your model detects?

**Expected Answer**: (1) Impaction — tooth trapped in bone/gum, (2) Caries — tooth decay/cavity, (3) Deep Caries — advanced decay reaching the pulp, (4) Periapical Lesion — infection at the root tip.

---

### Q8. What does `data_type` mean in your label format?

**Expected Answer**: It indicates annotation completeness: 0 = only quadrant label (Part 1), 1 = full FDI number (Part 2), 2 = FDI + disease attributes (Part 3). Used to mask which losses are computed per sample.

---

### Q9. What is the difference between training and inference?

**Expected Answer**: Training = learning from labeled data (forward pass + loss computation + backward pass to update weights). Inference = using the trained model to make predictions on new, unseen images (forward pass only, no weight updates).

---

### Q10. What is a confidence threshold and what happens if you lower it?

**Expected Answer**: Minimum probability score to keep a detection. Default = 0.25. Lowering it → more detections (including weaker ones) → higher sensitivity but more false positives. Raising it → fewer detections → higher precision but may miss real teeth.

---

### Q11. What is NMS (Non-Maximum Suppression)?

**Expected Answer**: A post-processing step that removes duplicate overlapping detections. If two boxes overlap by more than the IoU threshold (0.45), the one with lower confidence is suppressed. Ensures each tooth has at most one bounding box.

---

### Q12. Why is your input image size 1280×640 instead of 640×640?

**Expected Answer**: Panoramic X-rays have a ~2:1 aspect ratio (wide jaw). Using 640×640 would either stretch/distort the image, crop half the teeth, or waste pixels on padding. 1280×640 preserves the natural proportions.

---

### Q13. What is transfer learning and why do you use COCO-pretrained weights?

**Expected Answer**: Transfer learning uses knowledge from one task (COCO object detection) to help another (dental X-ray detection). COCO-pretrained YOLOv8x already knows how to extract edges, textures, and detect objects — we just fine-tune it to detect teeth instead of cars/people.

---

### Q14. What is an epoch?

**Expected Answer**: One complete pass through the entire training dataset. If we have 1000 images and batch_size=4, one epoch = 250 batches (iterations). We train for 100 epochs in Phase 1.

---

### Q15. What is the learning rate?

**Expected Answer**: Controls how much model weights change per update step. Too high → overshoots optimal weights and oscillates. Too low → converges very slowly. Our Phase 1 uses lr=0.01 (training from scratch), Phase 2 uses lr=0.002 (fine-tuning — smaller steps to preserve learned knowledge).

---

### Q16. What is a loss function?

**Expected Answer**: A mathematical function that measures how wrong the model's predictions are compared to ground truth. The optimizer adjusts weights to minimize this value. We use CIoU loss for boxes, Cross-Entropy for classes, and BCE for disease attributes.

---

### Q17. What does "freezing" the backbone mean?

**Expected Answer**: Setting `requires_grad = False` for all backbone parameters so they don't update during training. We freeze the backbone in Phase 2b (attribute head training) to preserve detection quality while only training the small disease classification heads.

---

### Q18. What is augmentation and why is it important?

**Expected Answer**: Artificially transforming training images (flip, rotate, brightness change) to create more diverse training examples. Prevents overfitting on a small dataset (~2000 images). Key: our horizontal flip also remaps tooth class labels (Q1↔Q2, Q3↔Q4).

---

### Q19. What is the validation set used for?

**Expected Answer**: A held-out set of images (~100) NOT used for training. We evaluate the model on it after each epoch to monitor performance and detect overfitting. The best validation score determines which checkpoint is saved as "best.pt".

---

### Q20. What is mAP and how is it calculated?

**Expected Answer**: Mean Average Precision — the standard detection metric. For each class, compute the area under the precision-recall curve (= Average Precision). Then average all 32 per-class APs. mAP@0.5 uses IoU≥0.5 as the "correct detection" threshold.

---

## Level 2: MEDIUM (Technical Understanding & Reasoning)

These test whether you understand WHY specific techniques are used and how they work together.

---

### Q1. Explain CoordConv. Why can't standard convolution distinguish tooth 16 from tooth 36?

**Expected Answer**: Standard Conv2d is translation-invariant — it produces identical outputs regardless of spatial position. Tooth 16 (upper-right molar) and tooth 36 (lower-left molar) look nearly identical visually. CoordConv concatenates normalized x,y coordinate channels (-1 to +1) to the input, breaking translation invariance. Now the model learns "molar shape + upper-right position → class 5 (tooth 16)" vs "molar shape + lower-left position → class 21 (tooth 36)".

---

### Q2. What is the Feature Pyramid Network? Why do we need P3, P4, P5 — not just one scale?

**Expected Answer**: FPN creates multi-resolution feature maps by merging high-resolution (fine detail) features with low-resolution (semantic) features. We need multiple scales because: P3 (stride 8) detects small features like root tip lesions, P4 (stride 16) detects full teeth, P5 (stride 32) captures jaw-level context. A single scale would miss either small pathologies or large impacted teeth.

---

### Q3. Your model has a hierarchical loss. A batch contains images from all 3 parts. How does the loss function handle this?

**Expected Answer**: Each sample's `data_type` field controls which losses fire. The attribute BCE loss checks `mask = (data_types == 2)` — only Part 3 samples contribute to disease loss. Classification loss uses full 32-class CE for data_type≥1, but groups into 4 quadrants for data_type=0. Bbox loss fires for all samples. This prevents the model from learning wrong disease labels from data that has no disease annotations.

---

### Q4. Why is pseudo labeling necessary? What happens without it?

**Expected Answer**: Part 3 only labels diseased teeth. Without pseudo labels, the 25+ healthy teeth in each Part 3 image have no ground truth. During training, the model is penalized for correctly detecting them (counted as false positives). This trains the model to NOT detect healthy teeth — catastrophic for a dental system. Pseudo labeling adds healthy-tooth annotations so the model learns to detect ALL teeth.

---

### Q5. Explain the linear sum assignment. Give a concrete example of a conflict it resolves.

**Expected Answer**: Suppose detections A and B both have highest probability for FDI slot "tooth 16". Without post-processing, both get labeled as tooth 16 (duplicate). The Hungarian algorithm builds a cost matrix (32 slots × N detections), minimizes total cost, and assigns each detection to a unique FDI slot. Detection A → tooth 16 (highest prob), Detection B → tooth 17 (second-highest prob). No duplicates.

---

### Q6. Why do you train in multiple phases instead of training everything end-to-end?

**Expected Answer**: (1) Phase 1 needs no disease labels (Part 1+2 only) — learns basic detection. (2) Between phases, pseudo labeling fills in healthy-tooth annotations. (3) Phase 2b trains attribute heads with frozen backbone — prevents disease loss from corrupting detection. (4) Phase 3 trains hybrid components with everything else frozen — prevents new modules from destabilizing earlier training. Progressive training = stability.

---

### Q7. What is pos_weight in BCE loss? Why is deep caries pos_weight=8.0 but caries=3.0?

**Expected Answer**: pos_weight upweights false negatives (missed diseases). Deep caries has ~8% prevalence (very rare), so the model would rarely encounter positive examples and collapse to "always predict healthy." pos_weight=8 means missing a deep caries case costs 8× more than a false alarm. Caries (25% prevalence) is more common, so a lower pos_weight=3 suffices to prevent collapse.

---

### Q8. Explain the Swin Transformer's shifted window mechanism. Why not use regular global attention?

**Expected Answer**: Regular global attention is O(N²) — at P5 with 800 tokens it's manageable, but doesn't scale. Swin divides the feature map into fixed 4×4 windows and computes attention within each window (O(16²)=O(256) per window). Problem: no cross-window communication. Solution: alternating shifted windows — the second block shifts windows by half the window size, so tokens that were at one window's edge now share a window with tokens from the adjacent window. This achieves global information flow across the entire feature map without O(N²) cost.

---

### Q9. In Phase 2b, you sample attribute logits at each GT tooth's center. Why not use global average pooling over the entire feature map?

**Expected Answer**: Global average pooling gives one prediction per IMAGE, not per TOOTH. If an image has 28 teeth and 2 have caries, global pooling gives "this image has caries" — the model learns to predict disease for every tooth in the image. Per-tooth sampling (at cx,cy on the feature map) gives predictions for each specific tooth — matching what inference does (sample at each detected box center). This fixed a critical bug where the model predicted every tooth as diseased.

---

### Q10. What is the purpose of the quadrant auxiliary head in the hybrid model?

**Expected Answer**: It provides an extra gradient signal for spatial/quadrant awareness. During post-processing, confident quadrant predictions modify the linear sum assignment cost matrix — penalizing cross-quadrant FDI assignments. Example: if the quadrant head says "this tooth is definitely in Q2 (95%)", then assigning it to any Q1/Q3/Q4 slot gets a penalty. This resolves midline teeth confusion where tooth 11 and 21 are adjacent.

---

### Q11. Why is CLAHE applied randomly (50% probability) during training instead of always?

**Expected Answer**: (1) Not all X-rays benefit from CLAHE — some already have good contrast. Applying it always could over-sharpen and amplify noise. (2) Stochastic application acts as augmentation — the model sees both enhanced and original versions, learning to handle both. (3) At inference, CLAHE is NOT applied (deterministic pipeline), so the model must work without it too.

---

### Q12. Why do you use CosineAnnealingLR instead of ReduceLROnPlateau?

**Expected Answer**: CosineAnnealingLR decays smoothly from lr_max to lr_min over a fixed number of epochs — predictable, no hyperparameter tuning for "patience" or "factor." ReduceLROnPlateau waits for a plateau then drops lr abruptly — can cause sudden jumps that destabilize training. Cosine annealing is the standard in YOLO literature and gives the model a smooth transition from exploration to refinement.

---

### Q13. The attribute heads use `nn.init.constant_(self.out.bias, -math.log(99))`. What does this do and why?

**Expected Answer**: At initialization, sigmoid(-log(99)) ≈ 0.01. This means the model starts by predicting ~1% probability of disease for every tooth — matching the prior that most teeth are healthy. Without this, random initialization might produce 50% predictions, causing large initial losses and unstable training. This is a form of "prior-aware initialization."

---

### Q14. Explain why the backbone must be in `eval()` mode during Phase 2b, even though we're "training."

**Expected Answer**: BatchNorm layers behave differently in train vs eval mode. In train mode, BN normalizes using per-batch statistics (mean/var of current 4 images). With batch_size=4, these statistics are noisy. In eval mode, BN uses accumulated running statistics (stable). If the backbone is in train mode, FPN features fluctuate batch-to-batch → attribute heads learn from inconsistent features → poor generalization at inference (which always uses eval mode).

---

### Q15. What is the "flip mapping" augmentation? Why can't you use albumentations' HorizontalFlip directly?

**Expected Answer**: When you horizontally flip a dental X-ray, upper-right becomes upper-left (Q1↔Q2, Q3↔Q4). Standard augmentation libraries flip the image and bounding box x-coordinates (`cx_new = 1 - cx`) but DON'T change the class label. A tooth labeled as class 5 (FDI 16, upper-right first molar) must become class 13 (FDI 26, upper-left first molar) after flipping. Our custom augmentor remaps class IDs using the `FLIP_CLASS_TABLE`.

---

### Q16. Why do you use `F.scaled_dot_product_attention` in the cross-attention instead of manually computing attention?

**Expected Answer**: (1) It automatically uses Flash Attention when available on the GPU — computing attention in tiles without materializing the full N×N attention matrix. At P3 scale (12,800 tokens), the full matrix would be 12,800² × 4 bytes ≈ 650 MB. Flash Attention computes the same result in O(N) memory. (2) It's numerically more stable. (3) It handles the dropout parameter correctly in train vs eval mode.

---

### Q17. What is DFL (Distribution Focal Loss)? How is it different from L1/L2 box regression?

**Expected Answer**: L1/L2 loss predicts a single coordinate value — assumes box edges are deterministic. DFL predicts a probability distribution over discretized possible edge positions (e.g., 16 bins). This captures uncertainty at blurry boundaries (tooth root fading into bone). The model can say "60% at position 5, 30% at position 6" instead of committing to one value. The final coordinate is the expected value of the distribution.

---

### Q18. How does the severity loss convert binary DENTEX labels into 3-class severity targets?

**Expected Answer**: DENTEX provides binary flags (0/1). The `binary_to_severity()` function maps: impacted 1→severity 2 (always severe), caries 1 + deepcaries 0→severity 1 (mild), caries 1 + deepcaries 1→severity 2 (severe), lesion 1→severity 2 (severe). The key insight: "has_caries AND has_deepcaries" together indicate severe, while "has_caries alone" indicates mild. This derives ordinal severity from existing binary annotations.

---

### Q19. What is early stopping and why is patience set to 50 (not 10 or 100)?

**Expected Answer**: Early stopping halts training when validation mAP doesn't improve for `patience` epochs. Too low (10): stops prematurely — detection models often plateau for 20-30 epochs before breaking through. Too high (100): wastes compute and risks overfitting after the model has clearly stagnated. 50 = enough time for the cosine scheduler to reduce lr and potentially find a new minimum, but not so much that we overfit.

---

### Q20. Why is mosaic augmentation disabled? What would happen if you enabled it?

**Expected Answer**: Mosaic stitches 4 random images into a 2×2 grid, creating one composite image. For dental X-rays, this creates non-existent anatomy: jawbones from different patients that don't connect, teeth floating at boundaries between images. The model would learn spurious patterns at junction lines. Panoramic X-rays are one continuous anatomical view — mixing them destroys spatial coherence that the model (especially with CoordConv and Swin) relies on.

---

## Level 3: HARD (Deep Technical, Debugging, Design Tradeoffs)

These test whether you can debug issues, propose improvements, and defend design choices under pressure.

---

### Q1. SCENARIO: After Phase 2b training, your model predicts EVERY tooth as having caries. Attribute loss was decreasing normally. What went wrong and how do you fix it?

**Expected Answer**: This is the "global average pooling bug." If the attribute head training used global spatial average of the feature map (instead of per-tooth center sampling), the model learns "this IMAGE contains caries" → predicts caries for ALL spatial positions → every tooth gets flagged. Fix: sample attribute logits at each GT tooth's (cx, cy) position in the feature map, exactly as inference samples at each detected box center. This was an actual bug we fixed with per-tooth supervision.

---

### Q2. SCENARIO: Phase 2 reaches best mAP at epoch 2, then oscillates for 50 epochs before early stopping. What's the most likely cause and fix?

**Expected Answer**: Learning rate too high for fine-tuning. Phase 2 starts from Phase 1's well-trained weights. With lr=0.01 (same as Phase 1 from-scratch), the first update overshoots the good minimum → model quality drops → oscillates around the minimum without converging. Fix: use a lower lr for Phase 2 (we use 0.002) — small enough to preserve Phase 1 knowledge while allowing gentle improvement. This is standard fine-tuning practice.

---

### Q3. SCENARIO: You notice that the attribute heads work perfectly on the validation set but produce all-healthy predictions on new hospital X-rays. What could cause this domain gap?

**Expected Answer**: Likely a preprocessing mismatch. If Phase 2b training uses letterboxing but the hospital inference pipeline stretches images (or vice versa), the FPN features at each spatial position represent different anatomical structures. The attribute heads learned "at position (0.3, 0.5) in the feature map, this is where tooth 16's center falls" — but with different preprocessing, that position corresponds to a different tooth or empty space. Fix: ensure identical preprocessing (letterboxing with same target size) in both training and inference.

---

### Q4. If you could only keep ONE of your three hybrid improvements (Swin, Cross-Attention, Severity Head), which would you keep and why?

**Expected Answer**: The Swin Transformer (Improvement A). Reasoning: (1) Without global context, cross-attention has nothing meaningful to fuse — it needs the Swin output as K/V. (2) The severity head is just a richer prediction format — it doesn't fix the underlying FDI numbering conflicts that are the primary error source. (3) The Swin branch directly addresses the most impactful limitation (FDI conflicts between adjacent teeth) by providing jaw-wide context. The other improvements are additive on top of Swin.

---

### Q5. SCENARIO: You want to deploy this model on a mobile device (smartphone camera captures of printed X-rays). What changes would you make?

**Expected Answer**: (1) Knowledge distillation: train a YOLOv8s student from the YOLOv8x teacher. (2) Remove hybrid components (Swin + cross-attention) — too expensive for mobile. (3) Quantize the model (INT8 or FP16). (4) Reduce input resolution to 640×320. (5) Use ONNX/TFLite export for mobile inference engines. (6) Accept reduced accuracy for disease attributes — focus on detection + basic disease flagging. Trade-offs: ~5-10% mAP drop for 10-20× speedup.

---

### Q6. Explain exactly what happens when you call `model.attach_feature_hooks()`. What layers are hooked and why those specific ones?

**Expected Answer**: PyTorch forward hooks are registered on layers 15, 18, and 21 of the YOLOv8x model (accessed via negative indexing: -8, -5, -2 of the 23-layer sequential). These are the output layers of the FPN neck that feed into the Detect head. Layer 15 produces P3 (stride 8, 320 channels), layer 18 produces P4 (stride 16, 640 channels), layer 21 produces P5 (stride 32, 640 channels). The hooks capture the output tensor of each layer into `self._fpn_features` list without modifying the forward pass.

---

### Q7. SCENARIO: Your pos_weight is set to [5, 3, 8, 5] but the model still rarely predicts deep caries. What additional diagnostic steps would you take?

**Expected Answer**: (1) Check how many data_type=2 samples have has_deepcaries=1 in the training set — if only 10-20 positive examples exist, even high pos_weight can't overcome insufficient data. (2) Check if the loss is actually non-zero for deepcaries — add per-attribute loss logging. (3) Verify that labels_ext files correctly have the deepcaries column populated. (4) Check if pseudo-labeling accidentally marked diseased teeth as healthy (IoU threshold too low). (5) Consider increasing pos_weight further or using focal loss instead of BCE for extreme imbalance. (6) Data-level fix: oversample Part 3 images with deep caries.

---

### Q8. Why does your model use separate forward hooks instead of modifying YOLOv8's forward method directly?

**Expected Answer**: (1) Ultralytics' YOLO class rebuilds the model internally during .train() — any modifications to the model structure cause state_dict key mismatches. (2) Hooks are non-invasive — they capture outputs without changing the forward pass or model structure. (3) Hooks can be added/removed dynamically (training vs inference). (4) Preserves compatibility with ultralytics' built-in training, evaluation, and export functions. The alternative (subclassing DetectionModel) would break ultralytics' internal training loop and require maintaining a fork.

---

### Q9. SCENARIO: A colleague suggests replacing your 4-phase training with a single end-to-end training run. Argue for AND against this approach.

**Expected Answer**: 
**For**: (1) Simpler pipeline, fewer hyperparameters. (2) Joint optimization could find a better global minimum. (3) Fewer places for bugs (like the preprocessing mismatch between phases).
**Against**: (1) Can't do pseudo-labeling without a pre-trained Phase 1 detector — Part 3 healthy teeth remain unlabeled. (2) Disease loss on 2000 images would overwhelm the detection loss gradient for the 68M-param backbone — attribute heads are 1.4M params, so their gradient signal is tiny relative to the backbone. (3) BatchNorm instability: disease loss would cause different feature statistics early in training. (4) Ultralytics' training infrastructure handles Phase 1/2 automatically — deviating requires writing a full custom training loop from scratch.

---

### Q10. How does the quadrant penalty in linear sum assignment handle the case where a tooth is at the exact midline (between Q1 and Q2)?

**Expected Answer**: The penalty is confidence-gated. For midline teeth, the quadrant head's softmax output might be [0.45, 0.40, 0.08, 0.07] (near-uniform between Q1 and Q2). `max_quad_prob = 0.45`, `confidence_scale = (0.45 - 0.25) / (1.0 - 0.25) = 0.27`, `alpha = 0.5 × 0.27 = 0.13` — a very small penalty. The cost matrix is barely modified, so the assignment relies primarily on class probabilities rather than the quadrant constraint. This is by design — for ambiguous midline teeth, we don't force a quadrant.

---

### Q11. SCENARIO: You're training on a new dataset with 50,000 dental X-rays. What hyperparameters would you change and why?

**Expected Answer**: (1) Increase batch_size to 8 or 16 (more data → can afford larger batches for better gradient estimates). (2) Reduce epochs (50-70 instead of 100 — more data per epoch means faster convergence). (3) Reduce warmup_epochs (1.5 instead of 3 — less time needed to stabilize with more data). (4) Can increase patience to 30 (dataset is larger, improvements are more reliable). (5) Might reduce pos_weight if the larger dataset has better disease representation. (6) Could try smaller model (YOLOv8l) since more data reduces the need for large model capacity.

---

### Q12. Explain the memory implications of `.detach()` on FPN features in Phase 2b. What happens if you forget it?

**Expected Answer**: `.detach()` breaks the computational graph between the backbone and the attribute heads. Without it, when `loss.backward()` is called, PyTorch would try to backpropagate through the attribute heads → through the FPN features → all the way back to the backbone (even though backbone params have requires_grad=False). This requires storing the entire backbone's activation graph in memory (~3GB per image). With `.detach()`, the graph is cut — only the attribute heads' small activation graph is stored (~50MB). Without detach + with frozen backbone = wasted memory; without detach + with unfrozen backbone = accidental backbone updates.

---

### Q13. SCENARIO: Your model achieves mAP@0.5=0.85 but mAP@0.5:0.95=0.45. What does this gap tell you and how would you improve it?

**Expected Answer**: Large gap means boxes are approximately correct (≥50% overlap) but not precisely localized (fail at strict 70-95% IoU). Likely causes: (1) Blurry tooth boundaries on X-rays make precise localization inherently hard. (2) Adjacent teeth overlap in panoramic projections, making boundary ambiguity worse. Improvements: (1) Increase box loss weight (7.5 → 10) to emphasize precise localization. (2) Use higher resolution input (1920×960) for finer spatial detail. (3) Add a boundary refinement head (segmentation-aware). (4) DFL already helps capture boundary uncertainty — might increase its weight or number of discretization bins.

---

### Q14. Why does your cross-attention use Pre-LN (LayerNorm before attention) instead of Post-LN (after)?

**Expected Answer**: Pre-LN provides more stable gradients during training, especially for deeper transformer blocks. With Post-LN, the residual connection adds unnormalized attention output to the input — as depth increases, activations can grow unboundedly. Pre-LN normalizes before attention, keeping activations bounded. Empirically, Pre-LN transformers train more stably without careful learning rate warmup. This is especially important for our cross-attention because it's trained in Phase 3 with a relatively small learning rate — any instability would waste limited training epochs.

---

### Q15. SCENARIO: You notice that the Swin Transformer's `_build_blocks()` is called with different spatial sizes during training (due to varying image sizes after padding). Is this a problem?

**Expected Answer**: Yes, this is a potential issue. The Swin blocks include learnable parameters (attention biases) that depend on `input_resolution`. If resolution changes, blocks are rebuilt with new parameters on a different device, potentially losing previously learned weights. Our implementation handles this with lazy initialization (`_build_blocks` is called on first forward) and padding to window_size multiples. In practice, with fixed input size (1280×640 → P5 always 40×20), the resolution is constant during training. The padding logic (`pad_h`, `pad_w`) handles edge cases where H or W isn't divisible by window_size.

---

### Q16. Compare the computational cost of your cross-attention at P3 vs P5. Why might you skip P3 fusion in a resource-constrained setting?

**Expected Answer**: P3: 160×80 = 12,800 tokens. Cross-attention Q×K^T = 12,800² attention entries per head (before Flash Attention optimization). P5: 40×20 = 800 tokens → 800² = 640,000 entries. P3 is 256× more expensive than P5. With 4 heads and batch_size=4, P3 cross-attention dominates compute. In a resource-constrained setting, fusing only at P5 (where the global context naturally lives) captures most of the benefit. P3/P4 fusion adds incremental improvement (fine-grained disease features enriched with context) but at significant compute cost.

---

### Q17. SCENARIO: A reviewer argues "CoordConv is unnecessary because the Swin Transformer already provides position information." How do you respond?

**Expected Answer**: (1) CoordConv operates in the backbone (layers 0-9), providing position awareness from the earliest feature extraction layers. The Swin only processes P5 at the end — by then, position information must already be encoded for the backbone to produce useful features. (2) CoordConv is used in ALL phases including Phase 1 (baseline) where Swin doesn't exist. (3) They serve different purposes: CoordConv gives each individual pixel its absolute position (fine-grained, per-element). Swin gives relative position between patches (coarse-grained, relational). (4) They're complementary: CoordConv helps the backbone EXTRACT position-aware features, Swin helps REASON about global position relationships.

---

### Q18. Why does your project store both `labels/` (5-column) and `labels_ext/` (10-column) files? Why not just use 10-column everywhere?

**Expected Answer**: Ultralytics' built-in YOLO training reads standard 5-column labels (`class cx cy w h`). It doesn't know about our extra 5 columns (attributes + data_type). Phase 1 and Phase 2 use ultralytics' `.train()` method directly → need standard 5-column labels. Phase 2b and Phase 3 use our custom training loops that read 10-column labels_ext for attribute supervision. Maintaining both ensures compatibility: ultralytics pipeline uses labels/, our custom pipeline uses labels_ext/.

---

### Q19. SCENARIO: You add a 5th disease attribute (root fracture) to the model. List every file that needs changes and what changes.

**Expected Answer**: 
1. `config/dataset.yaml`: Add "has_root_fracture" to attribute_names
2. `config/model_config.yaml`: Update attributes list (add 5th entry with weight 8.0)
3. `src/data/preprocess.py`: Add mapping for the new disease category_id → attr index 4, extend label to 11 columns
4. `src/data/dataset.py`: Update `LABEL_COLS = 11`
5. `src/models/yolortho.py`: `ATTRIBUTE_NAMES` list → add "has_root_fracture", `num_attrs=5`
6. `src/models/heads.py`: `MultiAttributeHead` now creates 5 heads per scale (automatic from num_attrs=5)
7. `src/training/loss.py`: `AttributeBCELoss(num_attrs=5)`, update pos_weight list to length 5
8. `src/inference/postprocess.py`: `ATTR_NAMES` list, `ToothDetection` dataclass → add field
9. `src/models/hybrid_head.py`: `MultiScaleSeverityHead(num_attrs=5)`, update `binary_to_severity()` for new attribute
10. `src/utils/visualize.py`: Add color/abbreviation for the new disease

---

### Q20. SCENARIO: Your model works well on the DENTEX validation set (mAP=0.85) but a dentist says "it misses subtle early caries that I can clearly see." What is the fundamental limitation and how would you address it architecturally?

**Expected Answer**: 
**Fundamental limitation**: The model is trained with BINARY caries labels (present/absent). Early caries (demineralization, white spots) might not be annotated in DENTEX — only obvious cavities are labeled. The model never saw early caries during training, so it can't detect what it wasn't taught.

**Architectural solutions**: 
(1) The severity head (Improvement C) partially addresses this — severity level 1 (mild) is meant to capture early caries. But if training labels don't distinguish early from advanced, it can't learn the difference.
(2) Self-supervised pre-training on unlabeled dental X-rays (DINOv2) would help the backbone learn to represent subtle texture differences in enamel demineralization.
(3) CLAHE augmentation (Improvement D) improves visibility of early caries by enhancing local contrast.
(4) Higher input resolution (1920×960) would preserve more fine-grained texture information.
(5) Ultimately, the bottleneck is annotation quality — we need a dataset where dentists specifically label early caries separately from advanced caries. This is a data problem more than a model problem.

---

## Study Tips

1. **Read answers out loud** — viva is verbal; practice speaking technical language fluently
2. **Know the numbers** — lr values, loss weights, channel counts, epoch counts
3. **Always tie back to dental context** — don't just explain the technique; explain WHY it matters for teeth
4. **Prepare for "what if" follow-ups** — examiners often dig deeper after your initial answer
5. **Admit uncertainty gracefully** — "I haven't tested that specifically, but I believe..." is better than guessing confidently
6. **Draw diagrams** — keep the architecture diagram from Section 1 in your head; reference it when explaining data flow
