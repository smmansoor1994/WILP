"""
minimal_train_example.py — Minimal Phase 2b/3 Training Example

Shows the bare minimum code needed to train Phase 2b and Phase 3.
Useful if you want to customize the training pipeline.

Usage:
    python minimal_train_example.py
"""

import sys
from pathlib import Path

# ─── Setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import ARCHONTrainer, ARCHONHybridTrainer


# ═════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: Train Phase 2b Only (Attribute Heads)
# ═════════════════════════════════════════════════════════════════════════════

def train_phase2b_example():
    """
    Phase 2b: Fine-tune attribute heads on top of Phase 2 backbone.
    
    Freezes the detection backbone, trains only the 4 binary attribute heads
    (Impacted, Caries, Deep Caries, Lesion) for 100 epochs.
    
    Takes: ~30-45 minutes on CPU
    """
    print("=" * 80)
    print("PHASE 2b: Train Attribute Heads Only")
    print("=" * 80)
    
    # Load Phase 2 weights (detection backbone)
    base_weights = str(PROJECT_ROOT / "weights" / "archon_best.pt")
    config_path = "config/train_config.yaml"
    device = "cpu"
    
    # Create trainer (Phase 2)
    trainer = ARCHONTrainer(
        config_path=config_path,
        resume=False,
        device=device,
    )
    
    # Train attributes only (backbone frozen)
    trainer.train_attr_only(base_weights=base_weights)
    
    # Result saved to: outputs/runs/attr/weights/best.pt
    print("✓ Phase 2b done! Weights: outputs/runs/attr/weights/best.pt")


# ═════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: Train Phase 3 Only (Hybrid Components)
# ═════════════════════════════════════════════════════════════════════════════

def train_phase3_example():
    """
    Phase 3: Train hybrid components on top of Phase 2 backbone.
    
    Adds 3 new architectures:
      1. GlobalContextEncoder (Swin Transformer)
      2. MultiScaleFusion (Cross-Attention)
      3. HybridMultiTaskHead (3-level severity heads + quadrant auxiliary)
    
    Freezes detection backbone, trains only new components for 100 epochs.
    
    Takes: ~30-45 minutes on CPU
    """
    print("=" * 80)
    print("PHASE 3: Train Hybrid Components (Swin + Cross-Attention + Severity)")
    print("=" * 80)
    
    # Load Phase 2 weights (or Phase 2b if just trained)
    base_weights = str(PROJECT_ROOT / "outputs" / "runs" / "attr" / "weights" / "best.pt")
    config_path = "config/train_config.yaml"
    device = "cpu"
    
    # Create trainer (Hybrid)
    trainer = ARCHONHybridTrainer(
        config_path=config_path,
        resume=False,
        device=device,
    )
    
    # Train hybrid components
    trainer.train_hybrid(base_weights=base_weights)
    
    # Result saved to: outputs/runs/hybrid/weights/best.pt
    print("✓ Phase 3 done! Weights: outputs/runs/hybrid/weights/best.pt")


# ═════════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: Train Both Phase 2b → Phase 3 (Sequential)
# ═════════════════════════════════════════════════════════════════════════════

def train_both_example():
    """
    Sequential training: Phase 2b → Phase 3
    
    Phase 2b improves disease detection accuracy.
    Phase 3 adds architectural improvements on top.
    
    Total time: ~1.5-2 hours on CPU
    """
    print("=" * 80)
    print("PHASE 2b → PHASE 3 (Sequential Training)")
    print("=" * 80)
    
    config_path = "config/train_config.yaml"
    device = "cpu"
    
    # ─── Phase 2b ───────────────────────────────────────────────────────────
    print("\n[1/2] Training Phase 2b...")
    base_weights = str(PROJECT_ROOT / "weights" / "archon_best.pt")
    
    trainer_2b = ARCHONTrainer(config_path=config_path, resume=False, device=device)
    trainer_2b.train_attr_only(base_weights=base_weights)
    
    phase2b_weights = PROJECT_ROOT / "outputs" / "runs" / "attr" / "weights" / "best.pt"
    print(f"✓ Phase 2b done: {phase2b_weights}")
    
    # ─── Phase 3 ───────────────────────────────────────────────────────────
    print("\n[2/2] Training Phase 3...")
    trainer_3 = ARCHONHybridTrainer(config_path=config_path, resume=False, device=device)
    trainer_3.train_hybrid(base_weights=str(phase2b_weights))
    
    phase3_weights = PROJECT_ROOT / "outputs" / "runs" / "hybrid" / "weights" / "best.pt"
    print(f"✓ Phase 3 done: {phase3_weights}")
    
    # ─── Copy to weights/ folder ────────────────────────────────────────────
    import shutil
    shutil.copy2(phase2b_weights, PROJECT_ROOT / "weights" / "attr_best.pt")
    shutil.copy2(phase3_weights, PROJECT_ROOT / "weights" / "hybrid_best.pt")
    
    print("\n✓ All done! Weights copied to weights/ folder:")
    print(f"  - weights/attr_best.pt    (Phase 2b)")
    print(f"  - weights/hybrid_best.pt  (Phase 3)")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 80)
    print("MINIMAL ARCHON Phase 2b/3 Training Example")
    print("=" * 80 + "\n")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
    else:
        print("Usage: python minimal_train_example.py [2b|3|both]")
        print("  2b    → Train Phase 2b only (attribute heads)")
        print("  3     → Train Phase 3 only (hybrid components)")
        print("  both  → Train Phase 2b then Phase 3 sequentially")
        print("\nExample: python minimal_train_example.py both")
        sys.exit(1)
    
    if choice == "2b":
        train_phase2b_example()
    elif choice == "3":
        train_phase3_example()
    elif choice == "both":
        train_both_example()
    else:
        print(f"Invalid choice: {choice}")
        sys.exit(1)
