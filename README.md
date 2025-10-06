# AST2000 — Project Portfolio Snapshot

> **Disclosure:** This README was written with the help of ChatGPT. **No generative AI was used to create the code, analyses, or figures** in this repository.

A compact overview of my AST2000 course project work (UiO): orbital mechanics, navigation (stereographic attitude, Doppler/radial‑velocity, trilateration), transfer design, and simple propulsion modeling. The goal of this README is to present the repository clearly for **recruiters and hiring managers**.

---

## What this demonstrates

* **Numerical methods:** RK4, leapfrog, stability/energy checks, Kepler verification.
* **Physics & astro:** two‑/N‑body dynamics, transfer windows, Δv budgeting, spectral line → radial‑velocity.
* **Data & software:** clean NumPy/Matplotlib pipelines, reproducibility (fixed seeds/units), version control discipline.

---

## How to run (quick)

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U numpy scipy matplotlib numba
# optional: pandas jupyter tqdm pillow astropy
```

Then open the relevant folder and run the script/notebook, e.g.

```bash
cd del4
python run_*.py    # or open the .ipynb
```

---

## Repository layout (short)

```
AST2000Victor/
├─ del1 … del9/           # Course “Del” parts with code & figures
├─ Solutions/             # Working notes / drafts
├─ planet_table.txt       # Supplied system parameters
├─ partC.py               # Utilities for select sub‑tasks
└─ README.md
```

---

## Selected highlights

* Orbit simulation with conservation diagnostics and Kepler checks.
* Stereographic sky‑map tiling for attitude/orientation.
* Doppler pipeline: spectral shift → radial‑velocity curve.
* Positioning via trilateration.
* Idealized rocket model for Δv/fuel estimates.

---

## Reproducibility

* **Units:** SI (axes labeled); some tasks use µm/km where appropriate.
* **Seeds:** Fixed when randomness is used.
* **Parallelism:** Minor FP drift possible; aggregate metrics remain stable.

---

## Tech stack

Python (NumPy, SciPy, Matplotlib, Numba), Jupyter. Git for version control.

---

## License

MIT (unless specified otherwise in subfolders).
