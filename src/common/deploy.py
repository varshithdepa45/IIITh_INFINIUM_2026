"""Warehouse bundle for hosts that cannot run the pipeline (e.g. Streamlit Community Cloud).

  python run.py deploy-bundle        # local: export a self-contained copy of warehouse/maple.duckdb

The local warehouse is ~3 GB and is rebuilt from the 10 GB release by run.py, so it is never committed.
`build_bundle` copies every main / silver / gold object into a new DuckDB file as real tables (the
Parquet-backed `main.*` views and the `raw.*` / `hidden.*` views point at local release files, so they
cannot travel; `raw` and `hidden` are not used by the app). It also copies the data dictionary and
the prebuilt policy-doc index that Layer 2 reads from `data_dir`. The DuckDB file is gzipped and
split into parts below GitHub's 2 GiB release-asset limit, with a manifest of sha256 sums.

`ensure_warehouse` runs at app start: if `db_path` is missing and `warehouse_bundle_url` (config or
WAREHOUSE_BUNDLE_URL env var) is set, it downloads the parts, verifies them, and rebuilds `db_path`.
If the configured `data_dir` is absent it points `data_dir` at the downloaded release metadata.
"""
from __future__ import annotations
import gzip, hashlib, json, os, shutil, time, urllib.request
from pathlib import Path
import duckdb

PART_BYTES = 1_900_000_000          # below GitHub's 2 GiB per-asset limit
SCHEMAS = ("main", "silver", "gold")
MANIFEST = "manifest.json"
DICTIONARY = "DATA_DICTIONARY.xlsx"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _docs_cache(db_path: str) -> Path:
    # same naming rule as src/layer2/qa.py (QA.docs)
    return Path(db_path.replace(".duckdb", "_docs.pkl"))


def build_bundle(cfg: dict, out_dir: str = "dist/warehouse_bundle", log=print) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    db = out / "maple.duckdb"

    con = duckdb.connect(str(db))
    con.execute(f"SET memory_limit = '{cfg.get('silver', {}).get('memory_limit', '2GB')}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"ATTACH '{cfg['db_path']}' AS src (READ_ONLY)")
    objects = con.execute(f"""SELECT table_schema, table_name, table_type FROM information_schema.tables
                              WHERE table_catalog = 'src' AND table_schema IN {SCHEMAS}
                              ORDER BY table_schema, table_name""").fetchall()
    for schema in SCHEMAS:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    for schema, name, kind in objects:
        t1 = time.time()
        con.execute(f'CREATE TABLE {schema}."{name}" AS SELECT * FROM src.{schema}."{name}"')
        n = con.execute(f'SELECT count(*) FROM {schema}."{name}"').fetchone()[0]
        log(f"  {schema}.{name:<32} {kind:<10} {n:>12,} rows  {time.time()-t1:5.1f}s")
    con.execute("DETACH src")
    con.execute("CHECKPOINT")
    con.close()
    log(f"  bundle db: {db.stat().st_size / 1e9:.2f} GB")

    # Layer 2 metadata that lives in data_dir: dictionary (protected-attribute list) + policy-doc index
    hits = sorted(Path(cfg["data_dir"]).rglob(DICTIONARY))
    if not hits:
        raise FileNotFoundError(f"{DICTIONARY} not found under {cfg['data_dir']}")
    shutil.copy2(hits[0], out / DICTIONARY)
    from src.layer2.rag_docs import DocIndex
    ro = duckdb.connect(cfg["db_path"], read_only=True)
    docs = DocIndex(ro, cfg["data_dir"], str(out / "maple_docs.pkl"))
    ro.close()
    log(f"  policy-doc index: {len(docs.chunks)} chunks")

    # one gzip stream, split into parts below the asset limit
    splitter = _SplitWriter(out, "maple.duckdb.gz.part")
    with open(db, "rb") as src, gzip.GzipFile(fileobj=splitter, mode="wb", compresslevel=6) as gz:
        shutil.copyfileobj(src, gz, length=1 << 20)
    splitter.close()
    manifest = {"db_sha256": _sha256(db), "db_bytes": db.stat().st_size,
                "parts": [{"name": f.name, "sha256": _sha256(f), "bytes": f.stat().st_size}
                          for f in splitter.files],
                # dest is relative to the warehouse folder on the host
                "files": [{"name": DICTIONARY, "sha256": _sha256(out / DICTIONARY), "dest": f"release/{DICTIONARY}"},
                          {"name": "maple_docs.pkl", "sha256": _sha256(out / "maple_docs.pkl"),
                           "dest": "maple_docs.pkl"}]}
    db.unlink()
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"  {len(splitter.files)} part(s) + {MANIFEST} in {out}  ({(time.time()-t0)/60:.1f} min)")
    return manifest


class _SplitWriter:
    """Write-only file object that rolls over to a new part file every PART_BYTES."""
    def __init__(self, folder: Path, prefix: str):
        self.folder, self.prefix, self.files, self._f, self._n = folder, prefix, [], None, 0

    def _roll(self):
        if self._f: self._f.close()
        path = self.folder / f"{self.prefix}{len(self.files):02d}"
        self.files.append(path)
        self._f, self._n = open(path, "wb"), 0

    def write(self, b) -> int:
        b = memoryview(b)
        done = 0
        while done < len(b):
            if self._f is None or self._n >= PART_BYTES:
                self._roll()
            k = min(len(b) - done, PART_BYTES - self._n)
            self._f.write(b[done:done + k])
            self._n += k
            done += k
        return done

    def flush(self):
        if self._f: self._f.flush()

    def close(self):
        if self._f: self._f.close(); self._f = None


def _download(url: str, dest: Path, sha256: str | None = None, log=print) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    if sha256 and _sha256(tmp) != sha256:
        tmp.unlink()
        raise IOError(f"checksum mismatch for {url}")
    tmp.replace(dest)
    log(f"  downloaded {dest.name} ({dest.stat().st_size / 1e6:,.0f} MB)")


class _ConcatReader:
    """Read-only file object over several part files in order."""
    def __init__(self, paths: list[Path]):
        self._paths, self._f = list(paths), None

    def read(self, n: int = -1) -> bytes:
        while self._paths or self._f:
            if self._f is None:
                self._f = open(self._paths.pop(0), "rb")
            b = self._f.read(n)
            if b:
                return b
            self._f.close(); self._f = None
        return b""

    def __enter__(self): return self
    def __exit__(self, *exc):
        if self._f: self._f.close()


def ensure_warehouse(cfg: dict, log=print) -> dict:
    """Make cfg['db_path'] exist (download the bundle if needed). Returns cfg, possibly with
    data_dir pointed at the downloaded release metadata. Never touches an existing warehouse."""
    db = Path(cfg["db_path"])
    wh = db.parent
    meta_dir = wh / "release"
    if not db.exists():
        base = os.environ.get("WAREHOUSE_BUNDLE_URL") or cfg.get("warehouse_bundle_url")
        if not base:
            raise FileNotFoundError(f"{db} not found. Build it with `python run.py all` (see run.py), "
                                    "or set warehouse_bundle_url in config.yaml.")
        base = base.rstrip("/") + "/"
        wh.mkdir(parents=True, exist_ok=True)
        log(f"warehouse missing; downloading bundle from {base}")
        manifest = json.loads(urllib.request.urlopen(base + MANIFEST, timeout=60).read())
        for f in manifest["files"]:
            dest = wh / f["dest"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            _download(base + f["name"], dest, f["sha256"], log)
        parts = []
        for p in manifest["parts"]:
            part = wh / p["name"]
            _download(base + p["name"], part, p["sha256"], log)
            parts.append(part)
        tmp_db = db.with_suffix(".duckdb.tmp")
        with _ConcatReader(parts) as cat, gzip.GzipFile(fileobj=cat, mode="rb") as gz, open(tmp_db, "wb") as out:
            shutil.copyfileobj(gz, out, length=1 << 20)
        for part in parts:
            part.unlink()
        if _sha256(tmp_db) != manifest["db_sha256"]:
            tmp_db.unlink()
            raise IOError("rebuilt warehouse checksum mismatch")
        tmp_db.replace(db)
        log(f"warehouse ready: {db} ({db.stat().st_size / 1e9:.2f} GB)")
    if not Path(cfg["data_dir"]).exists() and (meta_dir / DICTIONARY).exists():
        cfg["data_dir"] = str(meta_dir)
    return cfg
