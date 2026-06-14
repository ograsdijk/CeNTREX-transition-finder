# CeNTREX Transition Finder

Utilities and a Streamlit app for finding CeNTREX TlF optical transitions.

The repository now supports a single precomputed-grid workflow centered on a
locally generated `transition_grid.pkl`. The older
`sorted_transitions`/`spectrum_match` path has been removed.

The current transition finder can use a precomputed electric-field grid. The grid
tracks state identities from zero field with `eigenshuffle`, filters forbidden E1
components by `mF` selection rules, and stores the allowed split components with
their `mF -> mF'` labels.

Hamiltonian energy differences are converted to IR laser frequencies during
precompute by dividing the UV transition frequency by `4`. The app's UV/IR toggle
then displays either the stored IR values or the corresponding UV values.

The app can filter lines by laser polarization after precompute. `All` is the
incoherent sum over `X`, `Y`, and `Z`. `X`, `Y`, and `Z` are linear lab-frame
polarizations; with a nonzero `Ez`, `Z` is aligned to the applied electric field.
`sigma+` and `sigma-` are coherent circular components about lab `z`.

## Setup

```powershell
uv sync
```

## Test

```powershell
uv run pytest
```

`pytest` is installed from the project's `dev` dependency group, while Docker
continues to use `uv sync --frozen --no-dev`.

## Precompute The Field Grid

Generate the default grid:

```powershell
uv run python compute_transitions.py
```

Defaults:

- `Ez = 0..500 V/cm` in `25 V/cm` steps
- ground `J = 0..12`
- excited `J = 1..10`
- E1 P/Q/R branches only
- allowed components satisfy `delta mF = -1, 0, +1`
- stored strengths for `All`, `X`, `Y`, `Z`, `sigma+`, and `sigma-`
- the excited-state construction basis includes two extra `J` levels above the
  requested labels, matching the reduced-Hamiltonian construction used for the
  tracked grid precompute

This writes the local generated artifact `transition_grid.pkl`. The full default
grid can take a long time; the script prints per-field timing and ETA
information. The grid file is intentionally ignored by git because the full
artifact can exceed GitHub's regular file-size limit.

For a faster full build on a multicore machine, use worker processes:

```powershell
uv run python compute_transitions.py --workers 8
```

`--workers` is the default worker count for parallel stages. Override individual
stages with `--e1-workers` or `--tracking-workers` when needed:

```powershell
uv run python compute_transitions.py --workers 8 --e1-workers 6 --tracking-workers 4
```

The setup line reports the effective worker counts for tracking and E1
construction.

If you already have a `transition_grid.pkl` from an older version, rebuild it. The
app rejects old grid metadata because older grids stored UV Hamiltonian differences
as if they were IR frequencies.

For a smaller test grid:

```powershell
uv run python compute_transitions.py --ez-max 50 --ground-j-max 2 --excited-j-max 2
```

## Run The App

```powershell
uv run streamlit run transition_finder.py
```

The app requires `transition_grid.pkl`. If the grid artifact is missing, generate it
first with `uv run python compute_transitions.py`.

## Docker

Build:

```powershell
docker build --progress plain --no-cache -t centrex_transition_finder .
```

The Docker build installs dependencies from `pyproject.toml` and `uv.lock`, then
runs the app from the current checkout. Provide `transition_grid.pkl` in the build
context if you want the container to start without generating the grid separately.

Run:

```powershell
docker run -d -p 8501:8501 --name centrex-transition-finder centrex_transition_finder
```

The Streamlit interface is available on port `8501`.
