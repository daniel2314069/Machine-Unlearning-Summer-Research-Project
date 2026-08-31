# Fail-closed GPU-server storage cleanup

This maintenance script only operates beneath:

```text
/home/tslin/Documents/jupyter_data/anLi
```

It refuses an unexpected repository/root path, symlinked root, dirty Git tree,
tracked-file overlap, live PID file, nested mount, hardlinked ScaPre artifact,
candidate-set change, missing retained result, or failed integrity report.
Cleanup records are saved outside both the repository and transfer staging area
under `/home/tslin/Documents/jupyter_data/anLi/storage_cleanup_records/`.

The three destructive scopes are deliberately separate. There is no
destructive `all` option.

## 1. Pull and preview

On the GPU server:

```bash
cd /home/tslin/Documents/jupyter_data/anLi/machine_unlearning
conda activate MU
git pull --ff-only origin main
maintenance/server_storage_cleanup/cleanup_server_storage.sh --preview all
```

Preview creates manifests and size summaries but deletes nothing.

## 2. Remove ScaPre generated runs

```bash
maintenance/server_storage_cleanup/cleanup_server_storage.sh \
  --apply scapre --confirm DELETE_SCAPRE_RUNS
```

This removes only the eight explicitly allowlisted ScaPre `runs/` roots after
verifying the retained tracked results and passed integrity reports.

## 3. Remove OCE generated binaries

```bash
maintenance/server_storage_cleanup/cleanup_server_storage.sh \
  --apply oce --confirm DELETE_OCE_GENERATED_BINARIES
```

This scans only eight allowlisted OCE experiment roots and deletes only
untracked raw generated images under a path component named `images/` or
`generated_images/`, plus binary weights with these extensions:

```text
.png .jpg .jpeg .gif .webp .bmp .tif .tiff
.safetensors .pt .pth .ckpt
```

Tracked files cause a hard failure. JSON checkpoint manifests, metrics, logs,
predictions, configs, result tables, reports, and the complete shared
`evaluation_references/` tree are not candidates. Contact sheets, review
sheets, plots, grids, and standalone qualitative figures outside raw image
directories are conservatively preserved even when they are untracked. If the
OCE failure-qualification raw image directories still exist, its existing
fixed-rule review archive must be present in `anLi/tmp` before this scope may
run. Therefore run the OCE scope before emptying `anLi/tmp`.

## 4. Empty the anLi transfer staging directory

Only after required archives have been downloaded and locally checksum-verified:

```bash
maintenance/server_storage_cleanup/cleanup_server_storage.sh \
  --apply tmp --confirm EMPTY_ANLI_TMP_AFTER_LOCAL_BACKUP
```

This empties the contents of
`/home/tslin/Documents/jupyter_data/anLi/tmp` but preserves the `tmp` directory
itself. It refuses symlinks, nested mounts, or a referenced live PID. The
cleanup record survives because it is written to the sibling
`storage_cleanup_records/` directory.

## Status and verification

Show the latest cleanup record:

```bash
maintenance/server_storage_cleanup/cleanup_server_storage.sh --status
```

After each destructive scope, rerun the read-only inventory:

```bash
maintenance/server_storage_inventory/audit_server_storage.sh
```

The cleanup runs synchronously and preserves its candidate manifest, SHA-256,
before/after Git state, timestamps, exit code, completion marker, and summary.
If SSH is interrupted, rerun the same scope after reconnecting; the script
recomputes candidates and will only continue from the remaining allowlisted
artifacts.
