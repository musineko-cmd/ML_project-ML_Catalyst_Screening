# DFT Workflow for Top HER Candidates

This project now includes `src/dft_vasp.py` for preparing VASP calculations for
the top generated HER candidates.

## Prepare Inputs

```bash
python src/dft_vasp.py prepare \
  --candidates results/rankings/generated_candidates_top10_near_zero.csv \
  --outdir dft_runs \
  --top-n 10
```

This creates:

- `dft_runs/H2/`: H2 reference calculation
- `dft_runs/rank_XX_*/clean/`: clean slab calculation
- `dft_runs/rank_XX_*/H_ads/`: H adsorbed slab calculation
- `dft_runs/submit_all.sh`: convenience SGE submission script
- `dft_runs/dft_case_metadata.csv`: case metadata

The generated VASP settings follow the tutorial style in
`/home/musineko/vasp_tutorial`: PBE, spin-polarized, `ENCUT=400`, slab k-points,
POTCAR files from `/home/shared/programs/vasp/vasp_pp/potpaw_PBE`, and SGE
`submit_vasp.sh` scripts. Use `--no-potcar` if you only want POSCAR/INCAR/KPOINTS
templates.

## Use Real Structures When Available

The generated candidate CSV does not contain atomic coordinates. If exact slab
POSCAR files are available, put them under a directory and pass it as
`--structures-dir`:

```bash
python src/dft_vasp.py prepare --structures-dir candidate_poscars --outdir dft_runs
```

Recognized POSCAR layouts include:

- `candidate_poscars/rank_01.vasp`
- `candidate_poscars/rank_01.POSCAR`
- `candidate_poscars/rank_01/POSCAR`
- `candidate_poscars/<sanitized_candidate_id>/POSCAR`

If no POSCAR is provided, the script creates a deterministic composition-matched
prototype slab from `surfaceComposition`. These prototype slabs are useful for
setting up calculations, but production DFT conclusions should use physically
validated structures.

## Run VASP

From the run directory:

```bash
cd dft_runs
qsub H2/submit_vasp.sh
qsub rank_01_*/clean/submit_vasp.sh
qsub rank_01_*/H_ads/submit_vasp.sh
```

Each generated `submit_vasp.sh` changes into its own calculation directory before
launching VASP, so it is safe to submit from `dft_runs_mp/` using the paths above.

or submit everything:

```bash
./submit_all.sh
```

## Summarize Adsorption Energies

After VASP finishes:

```bash
python src/dft_vasp.py summarize --outdir dft_runs
```

The script writes `dft_runs/dft_h_adsorption_summary.csv` with:

```text
dE_H = E(slab+H) - E(clean slab) - 1/2 E(H2)
dG_H = dE_H + 0.24 eV
```

The default `0.24 eV` is the common approximate `DeltaZPE - TDeltaS` correction
for HER H adsorption. Change it with `--free-energy-correction` if you compute
explicit vibrational corrections.
