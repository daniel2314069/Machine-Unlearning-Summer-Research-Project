# GPU-server full repository storage inventory

This maintenance tool performs a read-only inventory of every entry under:

```text
/home/tslin/Documents/jupyter_data/anLi/machine_unlearning
```

It includes hidden entries, `.git`, all OCE and ScaPre directories,
experiments, run outputs, generated images, checkpoints, logs, archives, and
local artifacts. Symlinks are recorded but never followed. The tool does not
delete, move, truncate, hash, or otherwise modify scanned content. Output is
written outside the repository under
`/home/tslin/Documents/jupyter_data/anLi/tmp/storage_inventory_runs/`.

The complete outputs include:

- `all_files.tsv.gz`: every non-directory entry, logical/allocated size,
  modification time, category, mode, hardlink count, and relative path.
- `all_directories.tsv.gz`: inclusive and direct directory rollups.
- `largest_files.csv` and `largest_directories.csv`: top 5,000 entries.
- `root_children.csv`, `category_summary.csv`, `extension_summary.csv`, and
  `age_summary.csv`: review-oriented summaries.
- `summary.md`, `summary.json`, scan errors, Git state, logs, and SHA-256 result
  manifest.

No output labels a path safe to delete. After the archive is returned, deletion
decisions must account for Git tracking, reproducibility references, active
jobs, unique results, and whether an artifact is regenerable.

## Server usage

```bash
cd /home/tslin/Documents/jupyter_data/anLi/machine_unlearning
conda activate MU
maintenance/server_storage_inventory/audit_server_storage.sh
```

The scan is detached; the terminal, SSH connection, and local computer may be
closed after launch succeeds. Before launch, the runner uses the active `MU`
Python for a lightweight syntax/parser preflight; its bytecode cache is written
under `anLi/tmp`, outside the repository being scanned.

Status:

```bash
maintenance/server_storage_inventory/audit_server_storage.sh --status
```

After status is `completed`, package the inventory:

```bash
maintenance/server_storage_inventory/audit_server_storage.sh --package
```

Packaging verifies every result hash without `jq`, then prints the archive
path, byte size, SHA-256, checksum-sidecar path, and exact `scp` command for
both files. It does not remove the scan outputs or any repository content.
