#!/bin/sh
set -eu

# --- Config ---
# required versions
NEMO="24.12"

# optional versions, leave empty for preinstalled versions
: "${NCCL:="v2.23.4-1"}"
: "${MODALITIES:="v0.4.0"}"
: "${TORCHTITAN:="main"}"
# : "${PYTORCH:="2.8.0"}"
: "${PYTORCH:="nightly"}"
: "${PYTHON:="3.12"}"
: "${FLASH_ATTENTION:=">=2.6.0"}"

# : "${NCCL:=}"
# : "${MODALITIES:=}"
# : "${PYTORCH:=}"
# : "${PYTHON:=}"
# : "${FLASH_ATTENTION:=}"

# --- Helpers ---
sanitize() {
  # Map any non [A-Za-z0-9._-] to underscores (POSIX tr)
  # shellcheck disable=SC2018,SC2019
  printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'
}

tag_or_stock() {
  # prints sanitized value or 'stock' if empty
  if [ -z "$1" ]; then
    printf 'stock'
  else
    sanitize "$1"
  fi
}

# --- Derived ---
BASE="nvcr.io/nvidia/nemo:${NEMO}"

# Pick runner
if command -v apptainer >/dev/null 2>&1; then
  RUNNER=apptainer
elif command -v singularity >/dev/null 2>&1; then
  RUNNER=singularity
else
  printf '%s\n' "Error: neither 'apptainer' nor 'singularity' found in PATH." >&2
  exit 1
fi
echo "Using runner: $RUNNER"

# Temp def file
DEF_FILE="$(mktemp -t Container.XXXXXX.def)"
cleanup() { rm -f "$DEF_FILE"; }
# trap cleanup EXIT INT TERM TODO Comment in again
echo "DEF_FILE is at: $DEF_FILE" # TODO remove

# Write shared helper functions inside container to the following path
get_nccl_version_f=/usr/local/bin/get_nccl_version.sh
# --- Generate def file (host expands values; conditions become literals) ---
cat >"$DEF_FILE" <<EOF
Bootstrap: docker
From: ${BASE}

%post
set -eu

# ---- Helper: get_nccl_version ----
# we write this function to a script so that we can also use it in %test later
cat > /usr/local/bin/get_nccl_version.sh <<'END_NCCL'
get_nccl_version() {
  for hdr in /usr/include/nccl.h /usr/include/nccl/nccl.h /usr/local/include/nccl.h; do
    if [ -f "\$hdr" ]; then
      awk '
        /^#define[ \t]+NCCL_MAJOR[ \t]/ {maj=\$3}
        /^#define[ \t]+NCCL_MINOR[ \t]/ {min=\$3}
        /^#define[ \t]+NCCL_PATCH[ \t]/ {pat=\$3}
        END { if (maj && min && pat) print maj "." min "." pat }
      ' "\$hdr"
      return
    fi
  done
  if ldconfig -p 2>/dev/null | grep -q libnccl.so; then
    lib=\$(ldconfig -p | awk '/libnccl.so/{print \$NF; exit}')
    strings "\$lib" 2>/dev/null | grep -Eo 'NCCL [0-9]+\.[0-9]+\.[0-9]+' | head -n1 | awk '{print \$2}'
  fi
}
END_NCCL
chmod +x $get_nccl_version_f

get_cuda_version() {
  # Returns CUDA_VERSION (empty if not found)
  local version=""
  if [ -f /usr/local/cuda/version.txt ]; then
    version=\$(grep -Eo '[0-9]+\.[0-9]+' /usr/local/cuda/version.txt | head -n1)
  elif command -v nvcc >/dev/null 2>&1; then
    version=\$(nvcc --version | grep -Eo 'release [0-9]+\.[0-9]+' | awk '{print \$2}')
  fi
  printf '%s' "\$version"
}

echo_installed_versions() {
  python --version 2>&1 | sed 's/^/Python: /'
  python -c 'import torch; print("PyTorch:", torch.__version__)' 2>/dev/null || echo "PyTorch: not installed"
  python -c 'import flash_attn; import sys; print("FlashAttention:", getattr(flash_attn, "__version__", "unknown"))' 2>/dev/null || echo "FlashAttention: not installed"
  python -c 'import torch; print("Torch NCCL:", getattr(getattr(torch, "cuda", None), "nccl", None) and torch.cuda.nccl.version() or "not available")' 2>/dev/null || echo "Torch NCCL: not available"
  pip list | awk '\$1 == "modalities" {print "Modalities:", \$2}' || echo "Modalities: not installed"
  pip list | awk '\$1 == "torchtitan" {print "torchtitan:", \$2}' || echo "torchtitan: not installed"
  . $get_nccl_version_f
  echo "System NCCL: \$(get_nccl_version || echo not installed)"
  echo "CUDA: \$(get_cuda_version || echo not installed)"
}

echo "=== Preinstalled versions ==="
echo_installed_versions
echo "================================="

# Portable CPU count
if CORES=\$(getconf _NPROCESSORS_ONLN 2>/dev/null); then :; else CORES=1; fi

# Figure out deb multiarch libdir (Ubuntu/Debian base in NV images)
LIBDIR="/usr/lib/\$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo x86_64-linux-gnu)"

# ---- Optional: NCCL from source ----
if [ -n "${NCCL}" ]; then
  echo "Building NCCL version ${NCCL}"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates build-essential \
    pkg-config dpkg-dev curl
  rm -rf /var/lib/apt/lists/*

  git clone https://github.com/NVIDIA/nccl.git /tmp/nccl
  cd /tmp/nccl
  git checkout "${NCCL}"
  # sm_80 + sm_90 are common for A100/H100; adjust as needed
  make -j"\$CORES" NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90"

  # Remove any existing NCCL libs (ignore if absent), then install
  rm -f "\$LIBDIR"/libnccl.so* "\$LIBDIR"/libnccl_static.a 2>/dev/null || :
  install -D -m 0644 build/lib/libnccl.so* "\$LIBDIR/"
  install -D -m 0644 build/lib/libnccl_static.a "\$LIBDIR/"
  if [ -d build/lib/pkgconfig ]; then
    install -d "\$LIBDIR/pkgconfig"
    cp -a build/lib/pkgconfig/* "\$LIBDIR/pkgconfig/"
  fi
  if [ -f build/include/nccl.h ]; then
    install -D -m 0644 build/include/nccl.h /usr/include/nccl.h
  fi
  cd /
  rm -rf /tmp/nccl
fi

# ---- uv + venv + PyTorch ----
mkdir -p /usr/local/uv
export UV_INSTALL_DIR=/usr/local/uv
export UV_PYTHON_INSTALL_DIR=/usr/local/uv/python
export UV_CACHE_DIR=/usr/local/uv/cache
curl -LsSf https://astral.sh/uv/install.sh | sh

# Preserve executables
chmod 755 /usr/local/uv/uv /usr/local/uv/uvx

# Reasonable perms for rest
find /usr/local/uv -type d -exec chmod 755 {} \;
find /usr/local/uv -type f ! -name 'uv' ! -name 'uvx' -exec chmod 644 {} \;

# Symlink (optional)
ln -sf /usr/local/uv/uv /usr/local/bin/uv

# Correct PATH (no /bin subdir)
export PATH="/usr/local/uv:/usr/local/bin:\${PATH}"
export UV_LINK_MODE=copy
export UV_VENV_CLEAR=1

# Create venv; if PYTHON is unset or empty use preinstalled default Python
if [ -n "${PYTHON}" ]; then
  uv venv --python="python${PYTHON}" /opt/modalities_venv
else
  uv venv /opt/modalities_venv
fi
. /opt/modalities_venv/bin/activate

uv pip install --upgrade pip setuptools wheel packaging ninja psutil

# ---- PyTorch (optional) ----
if [ -n "${PYTORCH}" ]; then
  cuda_version=\$(get_cuda_version)
  cuda_tag=\$(echo "\$cuda_version" | tr -d '.')
  if [ "${PYTORCH}" = "nightly" ]; then
    # Nightly builds have a different URL pattern
    uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu"\$cuda_tag"
  else
    uv pip install torch==${PYTORCH} --index-url https://download.pytorch.org/whl/cu"\$cuda_tag"
  fi
fi

# ---- Torchtitan ----
rm -rf /tmp/torchtitan
git clone https://github.com/pytorch/torchtitan.git /tmp/torchtitan
cd /tmp/torchtitan
git checkout "${TORCHTITAN}"
uv pip install .
cd /
rm -rf /tmp/torchtitan

# ---- Modalities (optional) ----
if [ -n "${MODALITIES}" ]; then
  rm -rf /tmp/modalities
  git clone https://github.com/Modalities/modalities.git /tmp/modalities
  cd /tmp/modalities
  git checkout "${MODALITIES}"
  uv pip install .
  cd /
  rm -rf /tmp/modalities
fi

# ---- FlashAttention (optional) ----
if [ -n "${FLASH_ATTENTION}" ]; then
  # Avoid building when prebuilt wheels exist
  uv pip install "flash-attn${FLASH_ATTENTION}" --no-build-isolation
fi

# ---- MPI + nccl-tests ----
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends build-essential git openmpi-bin libopenmpi-dev ca-certificates
rm -rf /var/lib/apt/lists/*

# Derive MPI_HOME robustly on Ubuntu
MPI_INC=\$( (mpicc --showme:incdirs 2>/dev/null || mpicc -showme:compile 2>/dev/null) | tr ' ' '\n' | sed -n 's/^-I//p' | head -n1 )
if [ -n "\$MPI_INC" ]; then
  MPI_HOME=\$(dirname "\$MPI_INC")
else
  MPI_HOME="/usr/lib/x86_64-linux-gnu/openmpi"
fi
export PATH="\$MPI_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$MPI_HOME/lib:\${LD_LIBRARY_PATH:-}"

git clone --depth=1 https://github.com/NVIDIA/nccl-tests.git /nccl-tests
cd /nccl-tests
make -j"\$CORES" MPI=1 MPI_HOME="\$MPI_HOME"

echo "=== Installed versions after updates ==="
echo_installed_versions
echo "================================="

%environment
export PATH="/opt/modalities_venv/bin:/usr/lib/x86_64-linux-gnu/openmpi/bin:\${PATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/openmpi/lib:\${LD_LIBRARY_PATH}"

%test
set -eu
echo "=== Running image self-test ==="
# Activate venv if it exists

fail=0

# Check if installed software versions match expected
# Python version check
if [ -n "${PYTHON}" ]; then
  got_py=\$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null || echo "missing")
  if [ "\$got_py" = "${PYTHON}" ]; then
    echo "Python OK (\$got_py)"
  else
    echo "Python MISMATCH (got \$got_py expected ${PYTHON})"
    fail=1
  fi
fi

# PyTorch import + version
if [ -n "${PYTORCH}" ]; then
  python - <<PY || { echo "PyTorch test failed"; fail=1; }

try:
    import torch
except Exception as e:
    print("PyTorch import failed:", e)
    raise SystemExit(1)
got=torch.__version__

pytorch_mismatch=False
if "${PYTORCH}" == "nightly":
  if not "dev" in got:
    pytorch_mismatch=True
elif not got.startswith("${PYTORCH}"):
  pytorch_mismatch=True

if pytorch_mismatch:
    print(f"PyTorch MISMATCH (got {got} expected ${PYTORCH})")
    raise SystemExit(1)

print("PyTorch OK", got)
PY
PYTORCH_EXIT=$?
fi

# FlashAttention import (+ version if exact spec)
if [ -n "${FLASH_ATTENTION}" ]; then
  python <<PY || { echo "FlashAttention test failed"; fail=1; }
import re
try:
    import flash_attn as fa
except Exception as e:
    print("FlashAttention import failed:", e)
    raise SystemExit(1)
exp="${FLASH_ATTENTION}"
got=getattr(fa,"__version__","unknown")
# If spec uses comparison (>=,==,<=) just report import success
if re.search(r'[<>=]', exp):
    print(f"FlashAttention OK (got {got}, spec {exp})")
else:
    if not got.startswith(exp):
        print(f"FlashAttention MISMATCH (got {got} expected prefix {exp})")
        raise SystemExit(1)
    print("FlashAttention OK", got)
PY
fi

# NCCL version (header or strings)
if [ -n "${NCCL}" ]; then
  . $get_nccl_version_f
  got_nccl=\$(get_nccl_version || echo "")
  # normalize expected NCCL version (strip leading 'v' and any suffix)
  expected_nccl=\$(echo "${NCCL}" | sed -E 's/^v?([0-9]+\.[0-9]+\.[0-9]+).*/\1/')
  if [ -n "\$got_nccl" ]; then
    if [ "\$got_nccl" = "\$expected_nccl" ]; then
      echo "NCCL OK (\$got_nccl)"
    else
      echo "NCCL MISMATCH (got \$got_nccl expected \$expected_nccl)"
      fail=1
    fi
  else
    echo "NCCL NOT FOUND (expected \$expected_nccl))"
    fail=1
  fi
fi

echo "=== Self-test complete ==="
exit \$fail

EOF

# --- Build ---
hash=$(sha256sum "$DEF_FILE" | awk '{print $1}')
OUT_DIR="output_${hash}"
mkdir -p "$OUT_DIR"
OUT_IMAGE="${OUT_DIR}/image_${hash}.sif"

# Write versions.txt with set versions
cat > "$OUT_DIR/versions_${hash}.txt" <<VERS
NEMO=${NEMO}
NCCL=${NCCL}
MODALITIES=${MODALITIES}
TORCHTITAN=${TORCHTITAN}
PYTORCH=${PYTORCH}
PYTHON=${PYTHON}
FLASH_ATTENTION=${FLASH_ATTENTION}
VERS

echo "Building image: $OUT_IMAGE"
"$RUNNER" build "$OUT_IMAGE" "$DEF_FILE" 2>&1 | tee -a "$OUT_DIR/build_${hash}.log"
printf '✅ Built %s\n' "$OUT_IMAGE"

# Move image and def file to output dir
mv "$OUT_IMAGE" "$OUT_DIR/"
mv "$DEF_FILE" "$OUT_DIR/image_${hash}.def"

echo "All outputs are in $OUT_DIR"