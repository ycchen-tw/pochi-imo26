# Local-storage migration, 2026-09-14

Only model storage remains under `/nfs/aimo/shared/fm-pochi/`. Source, runtime,
caches, mutable service state, AIMO3 data, logs and results use local `/data`.

The removed NFS content was copied to
`/data/home/ycc/work/fm-pochi/legacy/nfs-handoff-20260914T085939Z/` before cleanup.
Its schema-2 manifest verified 618 regular files (4,723,268,701 bytes), 79
directories and one symlink by content, size and target. The manifest SHA256 is
`bdf31ad1b477fa65d4b2791575e43d777545900f30d6da2bda6bbd2604c9ff68`.

The retained NFS model tree contained 85 regular files totaling 69,895,104,297
bytes before and after cleanup. This includes the two verified checkpoints and
the small Hugging Face cache metadata. No process held an open reference below
the FM-Pochi NFS root during migration.

Operational paths are defined in `env.sh`. `FM_POCHI_MODEL_ROOT` is the sole NFS
path; `FM_POCHI_RUNTIME`, `FM_POCHI_DATA_ROOT`, `FM_POCHI_STATE_ROOT`, cache and
run directories are local. The cleanup receipt and full per-file manifest are
kept in the local archive above.

The migration exposed absolute NFS paths in the extracted venv's `pyvenv.cfg`
and Python entrypoint shebangs. `relocate_runtime.py` now rewrites and validates
those paths idempotently; `setup.sh` runs it for existing and newly extracted
runtimes.
