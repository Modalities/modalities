#!/bin/sh
set -eu

# --- Config ---
# required versions
NEMO="24.12"

# optional versions, leave empty for preinstalled versions
: "${NCCL:="v2.23.4-1"}"
: "${MODS:=}"
: "${PYTORCH:=}"
: "${PYTHON:=}"
: "${FA:=}"

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
name="image"
for var in NEMO PYTORCH PYTHON NCCL MODS FA; do
  prefix=$(printf '%s' "$var" | tr 'A-Z' 'a-z')
  # Indirect expansion (POSIX): put value of $var into val (empty if unset)
  eval "val=\${$var-}"
  name="${name}_${prefix}-$(tag_or_stock "$val")"
done
OUT="${name}.sif"
echo "Building container: $OUT"

# Pick runner
if command -v apptainer >/dev/null 2>&1; then
  RUNNER=apptainer
elif command -v singularity >/dev/null 2>&1; then
  RUNNER=singularity
else
  printf '%s\n' "Error: neither 'apptainer' nor 'singularity' found in PATH." >&2
  exit 1
fi

# Temp def file
DEF_FILE="$(mktemp -t Container.XXXXXX.def)"
cleanup() { rm -f "$DEF_FILE"; }
trap cleanup EXIT INT TERM

# --- Generate def file (host expands values; conditions become literals) ---
cat >"$DEF_FILE" <<EOF
Bootstrap: docker
From: ${BASE}

%post
set -eu

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
  cd /
  rm -rf /tmp/nccl
fi

# ---- uv + venv + PyTorch ----
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:\$PATH"
export UV_LINK_MODE=copy
export UV_VENV_CLEAR=1

# Create venv; if PYTHON unset or empty use preinstalled default Python
if [ -n "${PYTHON-}" ]; then
  uv venv --python="python${PYTHON}" /opt/modalities_venv
else
  uv venv /opt/modalities_venv
fi

. /opt/modalities_venv/bin/activate
uv pip install --upgrade pip setuptools wheel packaging ninja
if [ -n "${PYTORCH}" ]; then
  # uv pip install "torch==${PYTORCH}"
  uv pip install torch==${PYTORCH} --index-url https://download.pytorch.org/whl/cu126

fi

# ---- Modalities (optional) ----
if [ -n "${MODS}" ]; then
  rm -rf /tmp/modalities
  git clone https://github.com/Modalities/modalities.git /tmp/modalities
  cd /tmp/modalities
  git checkout "${MODS}"
  uv pip install .
  cd /
  rm -rf /tmp/modalities
fi

# ---- FlashAttention (optional) ----
if [ -n "${FA}" ]; then
  # Avoid building when prebuilt wheels exist
  uv pip install "flash-attn${FA}" --no-build-isolation
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
EOF

# --- Build ---
# "$RUNNER" build "$OUT" "$DEF_FILE"
printf '✅ Built %s\n' "$OUT"