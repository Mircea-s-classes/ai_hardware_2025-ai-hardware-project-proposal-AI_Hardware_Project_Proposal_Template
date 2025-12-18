#!/bin/bash
# Script to copy RPi results to local machine
# Usage: ./copy_rpi_results.sh

set -e

RPI_IP="172.20.10.5"
RPI_USER="pi"
PROJECT_DIR="/Users/kvothesarchives/Workspace/courses/ECE4332-AI-HW/ai-hardware-project-proposal-visionmasters"
RESULTS_DIR="$PROJECT_DIR/results/rpi_results"

echo "📥 Copying RPi results to local machine..."
echo ""

# Create directories
echo "Creating directories..."
mkdir -p "$RESULTS_DIR/headless_metrics"
mkdir -p "$RESULTS_DIR/x11_display_metrics"
mkdir -p "$RESULTS_DIR/training_charts"
mkdir -p "$RESULTS_DIR/all_metrics"

# Copy headless metrics (best performance - 9.7 FPS)
echo ""
echo "📊 Copying headless metrics (pure AI performance)..."
scp $RPI_USER@$RPI_IP:~/emote_detector/results/metrics/metrics_20251217_213125.* "$RESULTS_DIR/headless_metrics/" 2>/dev/null || echo "⚠️  Headless metrics not found"

# Copy X11 display metrics (with display overhead - 2.5 FPS)
echo ""
echo "🖥️  Copying X11 display metrics..."
scp $RPI_USER@$RPI_IP:~/emote_detector/results/metrics/metrics_20251217_213704.* "$RESULTS_DIR/x11_display_metrics/" 2>/dev/null || echo "⚠️  X11 metrics not found"

# Copy ALL metrics files
echo ""
echo "📁 Copying all metrics files..."
scp -r $RPI_USER@$RPI_IP:~/emote_detector/results/metrics/* "$RESULTS_DIR/all_metrics/" 2>/dev/null || echo "⚠️  No metrics found"

# Copy training charts
echo ""
echo "📈 Copying training charts..."
scp -r $RPI_USER@$RPI_IP:~/emote_detector/results/charts "$RESULTS_DIR/training_charts/" 2>/dev/null || echo "⚠️  No charts found (may need to train on RPi first)"

# Copy the trained model
echo ""
echo "🧠 Copying trained model..."
scp $RPI_USER@$RPI_IP:~/emote_detector/pose_classifier_model.pkl "$RESULTS_DIR/" 2>/dev/null || echo "⚠️  Model not found"

# Copy pose data
echo ""
echo "💾 Copying training data..."
scp -r $RPI_USER@$RPI_IP:~/emote_detector/pose_data "$RESULTS_DIR/" 2>/dev/null || echo "⚠️  Training data not found"

echo ""
echo "✅ Copy complete!"
echo ""
echo "📁 Results saved to:"
echo "   $RESULTS_DIR"
echo ""
echo "📊 Key files:"
echo "   - Headless metrics (9.7 FPS):  headless_metrics/"
echo "   - X11 metrics (2.5 FPS):       x11_display_metrics/"
echo "   - All metrics:                 all_metrics/"
echo "   - Training charts:             training_charts/"
echo "   - Performance summary:         PERFORMANCE_SUMMARY.md"
echo ""
echo "🎓 Ready for presentation!"

