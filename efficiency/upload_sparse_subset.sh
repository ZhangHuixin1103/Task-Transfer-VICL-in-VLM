#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 USER@HOST [SAMPLES_PER_TASK] [REMOTE_PROJECT_ROOT]" >&2
  exit 2
fi

REMOTE="$1"
SAMPLES_PER_TASK="${2:-100}"
REMOTE_PROJECT_ROOT="${3:-/home/zhanghui21/Task-Transfer}"
if [[ ! "$SAMPLES_PER_TASK" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAMPLES_PER_TASK must be a positive integer." >&2
  exit 2
fi
if [[ ! "$REMOTE_PROJECT_ROOT" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
  echo "REMOTE_PROJECT_ROOT contains unsupported characters." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
"${PYTHON:-python3}" "${SCRIPT_DIR}/prepare_sparse_subset.py" \
  --samples-per-task "$SAMPLES_PER_TASK"

UPLOAD_LIST="${SCRIPT_DIR}/upload_subset_files.txt"
ARCHIVE="${SCRIPT_DIR}/upload_subset_${SAMPLES_PER_TASK}.tar.gz"
REMOTE_ARCHIVE="/tmp/vicl_upload_subset_${SAMPLES_PER_TASK}.tar.gz"

tar -C "$PROJECT_ROOT" -czf "$ARCHIVE" -T "$UPLOAD_LIST"
echo "Prepared $(du -h "$ARCHIVE" | awk '{print $1}') archive: $ARCHIVE"

scp "$ARCHIVE" "${REMOTE}:${REMOTE_ARCHIVE}"
ssh "$REMOTE" \
  "mkdir -p '$REMOTE_PROJECT_ROOT' && tar -xzf '$REMOTE_ARCHIVE' -C '$REMOTE_PROJECT_ROOT' && rm -f '$REMOTE_ARCHIVE'"

echo "Uploaded sparse data to ${REMOTE}:${REMOTE_PROJECT_ROOT}"
echo "Use --task-manifest efficiency/tasks_sparse.json for third-party runs."
