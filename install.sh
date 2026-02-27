#!/bin/bash
# AudioCapBench - Installation Script
# Uses uv for fast Python environment setup

set -e

echo "========================================"
echo "AudioCapBench Installation"
echo "========================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv version: $(uv --version)"

# Create virtual environment with Python 3.12
echo ""
echo "[1/3] Creating Python 3.12 virtual environment..."
uv venv .venv --python 3.12
source .venv/bin/activate
echo "  Python: $(python --version)"
echo "  Location: $(which python)"

# Install dependencies
echo ""
echo "[2/3] Installing dependencies..."
uv pip install \
    numpy \
    tqdm \
    datasets \
    soundfile \
    torchcodec \
    torch \
    pyyaml \
    openai \
    google-genai \
    nltk \
    rouge-score \
    bert-score

# Optional: aac-metrics (requires Java 1.8+)
echo ""
echo "[3/3] Checking optional dependencies..."

# Check Java for aac-metrics
if command -v java &> /dev/null; then
    JAVA_VER=$(java -version 2>&1 | head -1)
    echo "  Java found: $JAVA_VER"
    echo "  Installing aac-metrics..."
    uv pip install aac-metrics || echo "  Warning: aac-metrics install failed (Java may be incompatible)"
else
    echo "  Java not found. Skipping aac-metrics (will use fallback NLTK metrics)."
    echo "  To install later: apt install default-jre && pip install aac-metrics"
fi

# Download NLTK data
echo ""
echo "Downloading NLTK data..."
python -c "
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print('  NLTK data downloaded')
"

# Load credentials if available
if [ -f credentials.env ]; then
    echo ""
    echo "Loading credentials from credentials.env..."
    source credentials.env
    echo "  OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..."
    echo "  GEMINI_API_KEY: ${GEMINI_API_KEY:0:10}..."
    echo "  HF_TOKEN: ${HF_TOKEN:0:10}..."
fi

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To build the evaluation dataset:"
echo "  source credentials.env"
echo "  python -m audiocapbench.build_dataset --output-dir data/audio_caption"
echo ""
echo "To run evaluation:"
echo "  python -m audiocapbench.evaluate \\"
echo "      --provider openai --model gpt-4o-audio-preview \\"
echo "      --data-dir data/audio_caption \\"
echo "      --credentials credentials.env \\"
echo "      --max-samples 10 --no-aac-metrics"
echo ""
