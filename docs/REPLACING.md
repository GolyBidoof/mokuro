# Replacing the pip-installed mokuro with this fork

The mokuro OCR engine is a **Python package installed via pip** (it has no
npm counterpart). "Replacing mokuro" therefore always means making
`import mokuro` (and the `mokuro` CLI) resolve to this local checkout instead
of the PyPI wheel. All options below point Python at the **local checkout**;
the fork itself is never downloaded again.

**One-time clone** (skip if you already have a checkout):

```bash
git clone https://github.com/GolyBidoof/mokuro.git mokuro-fork
cd mokuro-fork
```

`comic_text_detector/` is vendored in this repository (it is not a git
submodule), so no submodule step is needed.

The fork's runtime dependencies (torch, manga-ocr, transformers, ...) must
already be installed in the target environment. Install them normally once.
The steps below only swap *which* mokuro code gets used.

## Option A: replace inside one virtualenv (recommended)

```bash
source /path/to/your-venv/bin/activate
pip uninstall -y mokuro
pip install -e /path/to/mokuro-fork --no-deps --no-build-isolation
```

`--no-deps` stops pip from fetching anything from the network, and
`--no-build-isolation` reuses the already-installed setuptools (fully offline
install). Because the fork keeps the same distribution name (`mokuro`), pip
cleanly supersedes the previous install, with no leftover copies.

Verify from inside that venv:

```bash
mokuro --version          # -> 0.4.0b
python -c "import mokuro; print(mokuro.__file__)"   # -> .../mokuro-fork/mokuro/__init__.py
```

## Option B: replace machine-wide (your default python3)

Same idea, but for the interpreter your scripts use by default:

```bash
pip3 uninstall -y mokuro
pip3 install -e /path/to/mokuro-fork --no-deps --no-build-isolation
```

Every `python3` process on the machine now imports the fork, and the global
`mokuro` command runs it too. If pip refuses with an
`externally-managed-environment` error, use Option A in a venv instead.
Repeat Option A inside any other virtualenv that should use it; venvs do not
inherit global installs unless created with `--system-site-packages`.

## Option C: zero-install pointer file (library imports only)

If you only use mokuro as a **library** and never call the `mokuro` CLI, a
one-line `.pth` file in the interpreter's site-packages is enough; pip is
never involved:

```bash
python3 - <<'EOF'
import site
site_pkgs = site.getsitepackages()[0]
with open(f"{site_pkgs}/mokuro-fork.pth", "w") as f:
    f.write("/path/to/mokuro-fork\n")
print("wrote", f"{site_pkgs}/mokuro-fork.pth")
EOF
```

Any `import mokuro` under that interpreter now resolves to the fork. Caveats:

- It affects **imports only**. A previously installed `mokuro` console
  command still runs the old version (use Option B to replace the CLI too).
- Remove the `.pth` file before `pip install`-ing mokuro again; the old
  distribution is not uninstalled, so `pip list` shows both.

## Notes

- **mokuro-bridge**: the bridge already auto-uses a sibling `mokuro/`
  checkout and otherwise honours `MOKURO_REPO=/path/to/mokuro-fork`; no pip
  step is needed there at all.
- **npm**: there is no npm package for the mokuro OCR engine, so nothing to
  replace on that side. The web reader (reader.mokuro.app) is a hosted app
  and never runs OCR locally.
