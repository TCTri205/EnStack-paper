"""
Demo script to illustrate checkpoint behavior with crash scenario.
This helps understand what happens to model weights during crash and resume.
"""


class SimpleModel:
    """Simulates a model with trackable weights"""

    def __init__(self):
        self.batches_trained = []  # Track which batches this model has seen

    def train_batch(self, batch_id):
        """Simulate training on a batch"""
        self.batches_trained.append(batch_id)

    def save(self, path):
        """Simulate saving model to disk"""
        return {
            "batches_trained": self.batches_trained.copy(),
            "num_batches": len(self.batches_trained),
        }

    def load(self, checkpoint):
        """Simulate loading model from disk"""
        self.batches_trained = checkpoint["batches_trained"].copy()

    def __repr__(self):
        if len(self.batches_trained) <= 10:
            return f"Model(trained={self.batches_trained})"
        else:
            first = self.batches_trained[:3]
            last = self.batches_trained[-3:]
            return (
                f"Model(trained=[{first}...{last}], total={len(self.batches_trained)})"
            )


def demo_crash_and_resume():
    print("=" * 70)
    print("CHECKPOINT CRASH & RESUME DEMO")
    print("=" * 70)
    print()

    # Scenario parameters
    save_steps = 500
    crash_at_step = 899
    total_batches = 1270

    print("Parameters:")
    print(f"  - save_steps = {save_steps}")
    print(f"  - crash_at_step = {crash_at_step}")
    print(f"  - total_batches = {total_batches}")
    print()

    # ========== SESSION 1: Training until crash ==========
    print("=" * 70)
    print("SESSION 1: Training until crash")
    print("=" * 70)
    print()

    model = SimpleModel()
    checkpoint_disk = None  # What's saved on disk

    for step in range(0, crash_at_step + 1):
        # Train this batch
        model.train_batch(step)

        # Save checkpoint every save_steps
        if (step + 1) % save_steps == 0:
            checkpoint_disk = model.save(f"checkpoint_step{step + 1}")
            print(f"Step {step}: Train batch {step}")
            print(f"  → SAVE checkpoint (step={step + 1})")
            print(
                f"  → Disk: {checkpoint_disk['num_batches']} batches (0-{checkpoint_disk['num_batches'] - 1})"
            )
            print(
                f"  → Memory: {len(model.batches_trained)} batches (0-{len(model.batches_trained) - 1})"
            )
            print()

    print(f"Step {crash_at_step}: Train batch {crash_at_step}")
    print()
    print("💥 CRASH!")
    print()

    print("Final state before crash:")
    print(
        f"  - Model in memory: {len(model.batches_trained)} batches (0-{len(model.batches_trained) - 1})"
    )
    print(
        f"  - Model on disk: {checkpoint_disk['num_batches']} batches (0-{checkpoint_disk['num_batches'] - 1})"
    )
    print(
        f"  - LOST: {len(model.batches_trained) - checkpoint_disk['num_batches']} batches!"
    )
    print()

    # ========== SESSION 2: Resume from checkpoint ==========
    print("=" * 70)
    print("SESSION 2: Resume from checkpoint")
    print("=" * 70)
    print()

    # Load checkpoint
    model_resumed = SimpleModel()
    model_resumed.load(checkpoint_disk)
    resume_step = checkpoint_disk["num_batches"]

    print("Load checkpoint:")
    print(f"  - step = {resume_step}")
    print(
        f"  - Model has: {len(model_resumed.batches_trained)} batches (0-{len(model_resumed.batches_trained) - 1})"
    )
    print()

    print("Resume training:")
    print(f"  - Skip batches: 0 to {resume_step - 1}")
    print(f"  - Train batches: {resume_step} to {total_batches - 1}")
    print()

    # Simulate resume (only show key steps)
    batches_retrained = []

    for step in range(resume_step, min(resume_step + 5, total_batches)):
        model_resumed.train_batch(step)

        # Check if this batch was already trained in Session 1
        if step <= crash_at_step:
            batches_retrained.append(step)
            marker = "⚠️ RE-TRAIN"
        else:
            marker = "✅ NEW"

        print(f"  Step {step}: Train batch {step} {marker}")

    # Fast forward to show the pattern
    if total_batches > resume_step + 5:
        print("  ...")
        print(f"  Step {crash_at_step}: Train batch {crash_at_step} ⚠️ RE-TRAIN")
        print(f"  Step {crash_at_step + 1}: Train batch {crash_at_step + 1} ✅ NEW")
        print("  ...")

        # Simulate training the rest
        for step in range(resume_step + 5, total_batches):
            model_resumed.train_batch(step)
            if step <= crash_at_step and step not in batches_retrained:
                batches_retrained.append(step)

    print()

    # ========== ANALYSIS ==========
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    print("Batches re-trained:")
    print(f"  - Count: {len(batches_retrained)}")
    print(f"  - Range: {min(batches_retrained)} to {max(batches_retrained)}")
    print("  - These batches were trained TWICE (Session 1 + Session 2)")
    print()

    print("Time wasted (estimate):")
    time_per_batch_sec = 3.22
    wasted_time_min = len(batches_retrained) * time_per_batch_sec / 60
    print(
        f"  - {len(batches_retrained)} batches × {time_per_batch_sec}s = {wasted_time_min:.1f} minutes"
    )
    print()

    print("Final model state:")
    print(f"  - Total batches in final model: {len(model_resumed.batches_trained)}")
    print(f"  - Expected: {total_batches}")
    print(
        f"  - Status: {'✅ CORRECT' if len(model_resumed.batches_trained) == total_batches else '❌ ERROR'}"
    )
    print()

    # Check for duplicates in final model
    unique_batches = set(model_resumed.batches_trained)
    if len(unique_batches) == len(model_resumed.batches_trained):
        print("  ✅ No duplicate batches in final model")
        print("     (Re-training overwrote the old weights)")
    else:
        print("  ❌ ERROR: Duplicate batches detected!")

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("✅ Final model is CORRECT:")
    print("   - All batches 0-1269 trained exactly once in final weights")
    print("   - No batches skipped")
    print("   - No duplicate batches in final model")
    print()
    print("⚠️  Trade-off:")
    print(f"   - {len(batches_retrained)} batches were computed TWICE")
    print(f"   - Wasted ~{wasted_time_min:.1f} minutes of GPU time")
    print("   - This is EXPECTED with checkpoint-based recovery")
    print()
    print("💡 To minimize wasted work:")
    print("   - Increase checkpoint frequency (e.g., save_steps=200)")
    print(f"   - Current: max {save_steps} batches wasted")
    print("   - With save_steps=200: max 200 batches wasted")
    print()


if __name__ == "__main__":
    demo_crash_and_resume()
