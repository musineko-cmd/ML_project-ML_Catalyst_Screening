"""Prepare and analyze VASP calculations for HER H adsorption candidates.

This script follows the lightweight ASE/VASP pattern used in
``/home/musineko/vasp_tutorial``: create POSCAR/INCAR/KPOINTS/submit scripts for
clean slabs, H-covered slabs, and H2 reference calculations. If real candidate
POSCAR files are available they should be supplied via ``--structures-dir``.
Otherwise the script creates deterministic composition-matched prototype slabs
from the candidate ``surface`` label. Those prototype slabs are useful for
screening setup, but should be replaced by physically validated structures for
production DFT conclusions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import molecule, surface
from ase.build import bulk as ase_bulk
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, covalent_radii
from ase.io import read, write


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "results/rankings/generated_candidates_top10_near_zero.csv"
DEFAULT_OUTDIR = ROOT / "dft_runs"
DEFAULT_SUBMIT_TEMPLATE = Path("/home/musineko/vasp_tutorial/submit_vasp.sh")
DEFAULT_PP_PATH = Path("/home/shared/programs/vasp/vasp_pp")
DEFAULT_PP_SET = "potpaw_PBE"

MAGNETIC_ELEMENTS = {"Fe", "Co", "Ni", "Mn", "Cr"}
NON_STRUCTURAL_ELEMENTS = {"H"}


@dataclass
class CandidateRecord:
    rank: int
    candidate_id: str
    formula: str
    surface: str
    facet: str
    site: str
    pred_dg_h: float | None


def parse_formula(formula: Any) -> dict[str, float]:
    """Parse a simple formula or surface label into element amounts."""

    text = str(formula or "")
    parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", text)
    comp: dict[str, float] = {}
    for element, amount in parts:
        if element not in atomic_numbers:
            continue
        comp[element] = comp.get(element, 0.0) + float(amount or 1.0)
    return comp


def surface_label_to_formula(surface_label: Any) -> str:
    """Extract a chemistry-like formula from generated surface labels.

    Labels such as ``MoB2-Ru-B01`` encode a MoB2 base with a Ru dopant/site
    suffix. We include the first dopant token but ignore bookkeeping tokens such
    as ``B01`` so that the prototype slab at least contains the chemically
    relevant elements.
    """

    label = str(surface_label or "")
    first = label.split("-")[0].replace("+1%", "")
    comp = parse_formula(first)
    tokens = label.split("-")[1:]
    for token in tokens:
        if token in atomic_numbers:
            comp[token] = comp.get(token, 0.0) + 1.0
            break
    if not comp:
        comp = parse_formula(label)
    return "".join(f"{el}{int(amount) if amount != 1 else ''}" for el, amount in comp.items())


def safe_name(text: str, max_len: int = 90) -> str:
    out = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text).strip("_")
    return out[:max_len] or "candidate"


def normalize_facet(value: Any) -> str:
    text = str(value or "111").strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit() and len(text) < 3:
        return text.zfill(3)
    return text


def facet_to_miller(facet: str) -> tuple[int, int, int]:
    text = normalize_facet(facet)
    digits = [int(ch) for ch in text if ch.isdigit()]
    if len(digits) >= 3:
        return digits[0], digits[1], digits[2]
    return 1, 1, 1


def lattice_constant_for(element: str) -> float:
    radius = covalent_radii[atomic_numbers[element]]
    if not np.isfinite(radius) or radius <= 0:
        return 4.0
    # FCC nearest-neighbor distance is a/sqrt(2). The scale factor gives a
    # conservative starting cell for mixed prototype slabs.
    return float(2.0 * math.sqrt(2.0) * radius * 1.12)


def host_element(composition: dict[str, float]) -> str:
    structural = {el: amount for el, amount in composition.items() if el not in NON_STRUCTURAL_ELEMENTS}
    if not structural:
        return "Pt"
    return max(structural.items(), key=lambda item: item[1])[0]


def integer_counts_for_slab(composition: dict[str, float], n_atoms: int) -> dict[str, int]:
    structural = {el: amount for el, amount in composition.items() if el not in NON_STRUCTURAL_ELEMENTS and amount > 0}
    if not structural:
        structural = {"Pt": 1.0}
    total = sum(structural.values())
    raw = {el: amount / total * n_atoms for el, amount in structural.items()}
    counts = {el: int(math.floor(value)) for el, value in raw.items()}
    missing = n_atoms - sum(counts.values())
    for element, _ in sorted(raw.items(), key=lambda item: item[1] - math.floor(item[1]), reverse=True)[:missing]:
        counts[element] += 1
    return counts


def assign_slab_symbols(slab: Atoms, composition: dict[str, float]) -> None:
    counts = integer_counts_for_slab(composition, len(slab))
    symbols: list[str] = []
    for element, count in counts.items():
        symbols.extend([element] * count)
    symbols = symbols[: len(slab)]

    # Deterministic spatial ordering avoids run-to-run symbol shuffling. Higher-z
    # atoms are assigned first so dopants are more likely to appear near surface.
    order = np.lexsort((slab.positions[:, 0], slab.positions[:, 1], -slab.positions[:, 2]))
    ordered_symbols = [None] * len(slab)
    for idx, symbol in zip(order, symbols):
        ordered_symbols[int(idx)] = symbol
    slab.set_chemical_symbols([symbol or host_element(composition) for symbol in ordered_symbols])


def constrain_bottom_layers(slab: Atoms, fixed_layers: int) -> None:
    if fixed_layers <= 0:
        return
    z_values = np.array(sorted(set(round(float(z), 3) for z in slab.positions[:, 2])))
    if z_values.size == 0:
        return
    cutoff = z_values[min(fixed_layers - 1, z_values.size - 1)] + 1e-3
    slab.set_constraint(FixAtoms(indices=[atom.index for atom in slab if atom.position[2] <= cutoff]))


def build_prototype_slab(
    surface_label: str,
    facet: str,
    size: tuple[int, int],
    layers: int,
    vacuum: float,
    fixed_layers: int,
) -> Atoms:
    formula = surface_label_to_formula(surface_label)
    composition = parse_formula(formula)
    host = host_element(composition)
    bulk = ase_bulk(host, "fcc", a=lattice_constant_for(host), cubic=True)
    slab = surface(bulk, facet_to_miller(facet), layers=layers, vacuum=vacuum)
    slab = slab.repeat((size[0], size[1], 1))
    slab.center(axis=2, vacuum=vacuum)
    assign_slab_symbols(slab, composition)
    set_initial_magnetic_moments(slab)
    constrain_bottom_layers(slab, fixed_layers)
    return slab


def set_initial_magnetic_moments(atoms: Atoms) -> None:
    magmoms = [5.0 if atom.symbol in MAGNETIC_ELEMENTS else 0.0 for atom in atoms]
    atoms.set_initial_magnetic_moments(magmoms)


def top_layer_indices(atoms: Atoms, tolerance: float = 0.35) -> list[int]:
    zmax = max(atom.position[2] for atom in atoms)
    return [atom.index for atom in atoms if atom.position[2] >= zmax - tolerance]


def adsorption_position(atoms: Atoms, site: str, height: float) -> np.ndarray:
    top = top_layer_indices(atoms)
    if not top:
        raise ValueError("Could not identify top-layer atoms for H adsorption.")
    xy_center = np.mean(atoms.get_positions()[:, :2], axis=0)
    top_sorted = sorted(top, key=lambda idx: np.linalg.norm(atoms[idx].position[:2] - xy_center))
    site_text = str(site or "top").lower()
    if "hollow" in site_text or "fcc" in site_text or "hcp" in site_text:
        chosen = top_sorted[: min(3, len(top_sorted))]
    elif "bridge" in site_text or "bri" in site_text:
        chosen = top_sorted[: min(2, len(top_sorted))]
    else:
        chosen = top_sorted[:1]
    base = np.mean([atoms[idx].position for idx in chosen], axis=0)
    base[2] = max(atom.position[2] for atom in atoms) + height
    return base


def add_h_adsorbate(slab: Atoms, site: str, height: float) -> Atoms:
    ads = slab.copy()
    ads += Atoms("H", positions=[adsorption_position(ads, site, height)])
    set_initial_magnetic_moments(ads)
    return ads


def h2_reference(cell: float) -> Atoms:
    atoms = molecule("H2")
    atoms.set_cell([cell, cell, cell])
    atoms.set_pbc(True)
    atoms.center()
    atoms.set_initial_magnetic_moments([0.0, 0.0])
    return atoms


def write_incar(path: Path, settings: dict[str, Any]) -> None:
    lines = ["INCAR created by ML_project_screening/src/dft_vasp.py"]
    for key, value in settings.items():
        if isinstance(value, bool):
            rendered = ".TRUE." if value else ".FALSE."
        else:
            rendered = str(value)
        lines.append(f" {key.upper()} = {rendered}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_kpoints(path: Path, kpts: tuple[int, int, int]) -> None:
    path.write_text(
        "KPOINTS created by ML_project_screening/src/dft_vasp.py\n"
        "0\n"
        "Monkhorst-Pack\n"
        f"{kpts[0]} {kpts[1]} {kpts[2]}\n"
        "0 0 0\n",
        encoding="utf-8",
    )


def species_order_for_vasp(atoms: Atoms) -> list[str]:
    """Return the species order used by ASE when POSCAR is written sorted."""

    return sorted(set(atoms.get_chemical_symbols()))


def potential_dir_for_element(pp_root: Path, element: str) -> Path:
    candidates = [
        pp_root / f"{element}_pv",
        pp_root / f"{element}_sv",
        pp_root / f"{element}_d",
        pp_root / element,
    ]
    for path in candidates:
        if (path / "POTCAR").exists():
            return path
    raise FileNotFoundError(f"No POTCAR found for {element} under {pp_root}")


def write_potcar(path: Path, atoms: Atoms, pp_path: Path, pp_set: str) -> None:
    pp_root = pp_path / pp_set
    if not pp_root.exists():
        raise FileNotFoundError(f"Pseudopotential set not found: {pp_root}")
    chunks = []
    used = []
    for element in species_order_for_vasp(atoms):
        pot_dir = potential_dir_for_element(pp_root, element)
        chunks.append((pot_dir / "POTCAR").read_bytes())
        used.append(f"{element}:{pot_dir.name}")
    path.write_bytes(b"".join(chunks))
    (path.parent / "POTCAR.species").write_text("\n".join(used) + "\n", encoding="utf-8")


def slab_incar(encut: int, nsw: int, ediffg: float, npar: int) -> dict[str, Any]:
    return {
        "encut": encut,
        "potim": 0.1,
        "sigma": 0.2,
        "ediff": "1.00e-05",
        "ediffg": ediffg,
        "algo": "fast",
        "gga": "PE",
        "prec": "med",
        "ibrion": 2,
        "isif": 0,
        "ismear": 1,
        "ispin": 2,
        "istart": 0,
        "npar": npar,
        "nsw": nsw,
        "nwrite": 1,
        "lcharg": False,
        "lvtot": False,
        "lwave": False,
        "lreal": "auto",
    }


def gas_incar(encut: int, nsw: int, ediffg: float, npar: int) -> dict[str, Any]:
    return {
        "encut": encut,
        "potim": 0.1,
        "sigma": 0.01,
        "ediff": "1.00e-05",
        "ediffg": ediffg,
        "algo": "fast",
        "gga": "PE",
        "prec": "med",
        "ibrion": 2,
        "isif": 0,
        "ismear": 0,
        "ispin": 2,
        "nupdown": 0,
        "istart": 0,
        "npar": npar,
        "nsw": nsw,
        "nwrite": 1,
        "lcharg": False,
        "lvtot": False,
        "lwave": False,
        "lreal": "auto",
    }


def copy_submit_script(run_dir: Path, template: Path, job_name: str) -> None:
    if template.exists():
        text = template.read_text(encoding="utf-8")
    else:
        text = """#!/bin/bash
#$ -V
#$ -S /bin/bash
#$ -N VASP_job
#$ -q all.q
#$ -pe mpi_48 48
#$ -j Y
#$ -o $JOB_NAME.o$JOB_ID
#$ -cwd

echo "Got $NSLOTS slots."
cat $TMPDIR/machines
export OMP_NUM_THREADS=1
cd $SGE_O_WORKDIR
VASP="/home/shared/programs/vasp/5.4.4+vtst+vaspsol+beef/vasp.5.4.4/bin/vasp_std"
mpirun -machinefile $TMPDIR/machines -n $NSLOTS $VASP
"""
    text = re.sub(r"#\$ -N .*", f"#$ -N {job_name}", text)
    # The tutorial submit script uses ``cd $SGE_O_WORKDIR``, which runs VASP in
    # the directory where qsub was called. Embed the calculation directory so
    # submissions like ``qsub H2/submit_vasp.sh`` still read the right INCAR.
    text = re.sub(r"cd \$SGE_O_WORKDIR", f"cd \"{run_dir.resolve()}\"", text)
    script = run_dir / "submit_vasp.sh"
    script.write_text(text, encoding="utf-8")
    script.chmod(0o755)


def write_vasp_case(
    run_dir: Path,
    atoms: Atoms,
    incar: dict[str, Any],
    kpts: tuple[int, int, int],
    submit_template: Path,
    job_name: str,
    write_potcar_file: bool,
    pp_path: Path,
    pp_set: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write(run_dir / "POSCAR", atoms, format="vasp", direct=True, sort=True, vasp5=True)
    write_incar(run_dir / "INCAR", incar)
    write_kpoints(run_dir / "KPOINTS", kpts)
    if write_potcar_file:
        write_potcar(run_dir / "POTCAR", atoms, pp_path, pp_set)
    copy_submit_script(run_dir, submit_template, job_name)


def load_candidates(path: Path, top_n: int) -> list[CandidateRecord]:
    df = pd.read_csv(path).head(top_n).copy()
    records = []
    for idx, row in df.iterrows():
        pred_dg_h = None
        for column in ("pred_dG_H", "pred_dG_H_ExtraTrees", "pred_dG_H_mu"):
            if column in row and pd.notna(row[column]):
                pred_dg_h = float(row[column])
                break
        records.append(
            CandidateRecord(
                rank=int(idx) + 1,
                candidate_id=str(row.get("candidate_id", row.get("candidate_key", idx))),
                formula=str(row.get("formula", row.get("chemicalComposition", ""))),
                surface=str(row.get("surface", row.get("surfaceComposition", ""))),
                facet=normalize_facet(row.get("normalized_facet", row.get("facet", "111"))),
                site=str(row.get("site", "top")),
                pred_dg_h=pred_dg_h,
            )
        )
    return records


def candidate_structure_path(structures_dir: Path | None, record: CandidateRecord) -> Path | None:
    if structures_dir is None:
        return None
    candidates = [
        structures_dir / f"rank_{record.rank:02d}.vasp",
        structures_dir / f"rank_{record.rank:02d}.POSCAR",
        structures_dir / f"rank_{record.rank:02d}" / "POSCAR",
        structures_dir / safe_name(record.candidate_id) / "POSCAR",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def prepare(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    submit_template = Path(args.submit_template)
    pp_path = Path(args.pp_path)
    records = load_candidates(Path(args.candidates), args.top_n)
    structures_dir = Path(args.structures_dir) if args.structures_dir else None

    write_vasp_case(
        outdir / "H2",
        h2_reference(args.gas_cell),
        gas_incar(args.encut, args.nsw, args.gas_ediffg, args.npar),
        (1, 1, 1),
        submit_template,
        "H2_ref",
        args.write_potcar,
        pp_path,
        args.pp_set,
    )

    metadata = []
    for record in records:
        case_dir = outdir / f"rank_{record.rank:02d}_{safe_name(record.formula + '_' + record.surface)}"
        structure_path = candidate_structure_path(structures_dir, record)
        if structure_path:
            clean = read(structure_path)
            clean.set_pbc([True, True, True])
            set_initial_magnetic_moments(clean)
            constrain_bottom_layers(clean, args.fixed_layers)
            source = str(structure_path)
        else:
            clean = build_prototype_slab(
                record.surface,
                record.facet,
                size=(args.slab_size[0], args.slab_size[1]),
                layers=args.layers,
                vacuum=args.vacuum,
                fixed_layers=args.fixed_layers,
            )
            source = "prototype_from_surface_composition"
        h_ads = add_h_adsorbate(clean, record.site, args.ads_height)

        write_vasp_case(
            case_dir / "clean",
            clean,
            slab_incar(args.encut, args.nsw, args.slab_ediffg, args.npar),
            tuple(args.kpts),
            submit_template,
            f"r{record.rank:02d}_clean",
            args.write_potcar,
            pp_path,
            args.pp_set,
        )
        write_vasp_case(
            case_dir / "H_ads",
            h_ads,
            slab_incar(args.encut, args.nsw, args.slab_ediffg, args.npar),
            tuple(args.kpts),
            submit_template,
            f"r{record.rank:02d}_Hads",
            args.write_potcar,
            pp_path,
            args.pp_set,
        )
        record_meta = asdict(record)
        record_meta.update(
            {
                "case_dir": str(case_dir),
                "structure_source": source,
                "surface_formula_used_for_prototype": surface_label_to_formula(record.surface),
                "clean_atoms": len(clean),
                "h_ads_atoms": len(h_ads),
            }
        )
        (case_dir / "metadata.json").write_text(json.dumps(record_meta, indent=2), encoding="utf-8")
        metadata.append(record_meta)

    pd.DataFrame(metadata).to_csv(outdir / "dft_case_metadata.csv", index=False)
    write_submit_all(outdir, metadata)
    print(f"Prepared {len(records)} candidate DFT cases under {outdir}")
    print(f"H2 reference: {outdir / 'H2'}")
    print(f"Metadata: {outdir / 'dft_case_metadata.csv'}")


def write_submit_all(outdir: Path, metadata: list[dict[str, Any]]) -> None:
    lines = ["#!/bin/bash", "set -euo pipefail", f"cd \"{outdir.resolve()}\"", "qsub H2/submit_vasp.sh"]
    for row in metadata:
        case = Path(row["case_dir"]).name
        lines.append(f"qsub {case}/clean/submit_vasp.sh")
        lines.append(f"qsub {case}/H_ads/submit_vasp.sh")
    script = outdir / "submit_all.sh"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)


def parse_final_energy(run_dir: Path) -> float | None:
    outcar = run_dir / "OUTCAR"
    if outcar.exists():
        energy = None
        for line in outcar.read_text(errors="ignore").splitlines():
            if "free  energy   TOTEN" in line:
                parts = line.split()
                try:
                    energy = float(parts[4])
                except (IndexError, ValueError):
                    continue
        if energy is not None:
            return energy
    oszicar = run_dir / "OSZICAR"
    if oszicar.exists():
        energy = None
        for line in oszicar.read_text(errors="ignore").splitlines():
            match = re.search(r"E0=\s*([-+0-9.Ee]+)", line)
            if match:
                energy = float(match.group(1))
        return energy
    return None


def parse_last_ionic_step(run_dir: Path) -> int | None:
    oszicar = run_dir / "OSZICAR"
    if not oszicar.exists():
        return None
    step = None
    for line in oszicar.read_text(errors="ignore").splitlines():
        match = re.match(r"\s*(\d+)\s+F=", line)
        if match:
            step = int(match.group(1))
    return step


def parse_run_status(run_dir: Path) -> dict[str, Any]:
    outcar = run_dir / "OUTCAR"
    text = outcar.read_text(errors="ignore") if outcar.exists() else ""
    return {
        "energy_eV": parse_final_energy(run_dir),
        "converged": "reached required accuracy" in text,
        "finished": "General timing" in text,
        "last_ionic_step": parse_last_ionic_step(run_dir),
    }


def summarize(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    metadata_path = outdir / "dft_case_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}; run prepare first.")
    h2_status = parse_run_status(outdir / "H2")
    h2_energy = h2_status["energy_eV"]
    rows = []
    status_rows = []
    for _, row in pd.read_csv(metadata_path).iterrows():
        case_dir = Path(row["case_dir"])
        clean_status = parse_run_status(case_dir / "clean")
        h_ads_status = parse_run_status(case_dir / "H_ads")
        e_clean = clean_status["energy_eV"]
        e_h_ads = h_ads_status["energy_eV"]
        d_e_h = None
        d_g_h = None
        if e_clean is not None and e_h_ads is not None and h2_energy is not None:
            d_e_h = e_h_ads - e_clean - 0.5 * h2_energy
            d_g_h = d_e_h + args.free_energy_correction
        reliable = bool(
            d_g_h is not None
            and clean_status["converged"]
            and h_ads_status["converged"]
            and h2_status["converged"]
            and clean_status["finished"]
            and h_ads_status["finished"]
            and h2_status["finished"]
        )
        if reliable:
            status = "complete_converged"
        elif e_clean is None and e_h_ads is None:
            status = "not_available"
        else:
            status = "partial_or_unconverged"
        rows.append(
            {
                **row.to_dict(),
                "E_clean_eV": e_clean,
                "E_H_ads_eV": e_h_ads,
                "E_H2_eV": h2_energy,
                "dE_H_eV": d_e_h,
                "free_energy_correction_eV": args.free_energy_correction,
                "dG_H_eV": d_g_h,
            }
        )
        status_rows.append(
            {
                "rank": row.get("rank"),
                "candidate_id": row.get("candidate_id"),
                "ml_pred_dG_H_eV": row.get("pred_dg_h"),
                "clean_converged": clean_status["converged"],
                "h_ads_converged": h_ads_status["converged"],
                "h2_converged": h2_status["converged"],
                "clean_finished": clean_status["finished"],
                "h_ads_finished": h_ads_status["finished"],
                "h2_finished": h2_status["finished"],
                "clean_last_ionic_step": clean_status["last_ionic_step"],
                "h_ads_last_ionic_step": h_ads_status["last_ionic_step"],
                "E_clean_eV": e_clean,
                "E_H_ads_eV": e_h_ads,
                "E_H2_eV": h2_energy,
                "dE_H_eV": d_e_h,
                "dG_H_eV": d_g_h,
                "reliable_dG_H": reliable,
                "status": status,
            }
        )
    summary = pd.DataFrame(rows)
    summary_path = outdir / "dft_h_adsorption_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved DFT adsorption summary -> {summary_path}")
    status_summary = pd.DataFrame(status_rows)
    status_path = outdir / "dft_h_adsorption_status_summary.csv"
    status_summary.to_csv(status_path, index=False)
    print(f"Saved convergence-aware DFT status -> {status_path}")
    converged_path = outdir / "dft_h_adsorption_converged_only.csv"
    status_summary[status_summary["reliable_dG_H"]].to_csv(converged_path, index=False)
    print(f"Saved reliable converged DFT results -> {converged_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare/analyze VASP DFT calculations for generated HER candidates.")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Create VASP input directories for H2, clean slabs, and H adsorbed slabs.")
    prep.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    prep.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    prep.add_argument("--top-n", type=int, default=10)
    prep.add_argument("--structures-dir", default="", help="Optional directory containing real POSCAR files for candidates.")
    prep.add_argument("--submit-template", default=str(DEFAULT_SUBMIT_TEMPLATE))
    prep.add_argument("--pp-path", default=str(DEFAULT_PP_PATH), help="Root containing VASP PAW potential sets.")
    prep.add_argument("--pp-set", default=DEFAULT_PP_SET, help="PAW set directory under --pp-path.")
    prep.add_argument("--no-potcar", dest="write_potcar", action="store_false", help="Skip POTCAR generation.")
    prep.set_defaults(write_potcar=True)
    prep.add_argument("--slab-size", type=int, nargs=2, default=[3, 3], metavar=("NX", "NY"))
    prep.add_argument("--layers", type=int, default=4)
    prep.add_argument("--fixed-layers", type=int, default=2)
    prep.add_argument("--vacuum", type=float, default=12.0)
    prep.add_argument("--ads-height", type=float, default=1.1)
    prep.add_argument("--gas-cell", type=float, default=20.0)
    prep.add_argument("--kpts", type=int, nargs=3, default=[3, 3, 1])
    prep.add_argument("--encut", type=int, default=400)
    prep.add_argument("--nsw", type=int, default=200)
    prep.add_argument("--slab-ediffg", type=float, default=-0.05)
    prep.add_argument("--gas-ediffg", type=float, default=-0.01)
    prep.add_argument("--npar", type=int, default=2)
    prep.set_defaults(func=prepare)

    summ = sub.add_parser("summarize", help="Parse completed VASP energies and compute dG_H.")
    summ.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    summ.add_argument(
        "--free-energy-correction",
        type=float,
        default=0.24,
        help="Approximate DeltaZPE - TDeltaS correction for H* vs 1/2 H2 in eV.",
    )
    summ.set_defaults(func=summarize)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
