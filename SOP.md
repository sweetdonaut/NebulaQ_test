# Nebula SEM Simulator — Quick Deploy SOP

Tested: 2026-03-23 on Ubuntu 24.04, g++ 13.3.0, cmake 3.28.3, CUDA 13.1, HDF5 1.10.10, Python 3.12

## Prerequisites

```bash
sudo apt install -y build-essential gfortran cmake git libhdf5-dev python3
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
```

CUDA (optional, for GPU): install from https://developer.nvidia.com/cuda-downloads, then:
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Directory Layout

```
NebulaQ_test/
├── nebula/
│   ├── nebula-master/
│   │   ├── Nebula-master.zip
│   │   └── elsepa-2020/
│   │       └── elsepa-2020.zip
│   └── cstool-master/
│       └── cstool-master.zip
├── cstool-env/                    # (auto-generated)
└── run-simulation/                # (auto-generated)
```

Place the three zip files in this structure before starting.

---

## Step 1: ELSEPA 2020

```bash
cd nebula/nebula-master/elsepa-2020
unzip -o elsepa-2020.zip
cd elsepa-2020
gfortran -O elscata.f -o elscata
gfortran -O elscatm.f -o elscatm
```

> **Note:** Only pass the main `.f` file. `elscata.f` already INCLUDEs `radial.f` and `elsepa2020.f`.
> Old docs say `gfortran -O elscata.f radial.f elsepa2020.f` — this fails on GCC 10+ (`-fno-common` default causes multiple definition errors).

Verify:
```bash
./elscata < elscata.in    # should produce dcs_*.dat files
```

## Step 2: Nebula

```bash
cd nebula/nebula-master
unzip -o Nebula-master.zip
```

**Patch CMakeLists.txt** (required for CUDA 12+):
```bash
sed -i 's/CMAKE_CXX_STANDARD 14/CMAKE_CXX_STANDARD 17/' Nebula-master/CMakeLists.txt
sed -i 's/CMAKE_CUDA_STANDARD 14/CMAKE_CUDA_STANDARD 17/' Nebula-master/CMakeLists.txt
```

Build:
```bash
cd Nebula-master
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

Verify:
```bash
./bin/nebula_gpu       # "This is Nebula version 1.0.2"
./bin/nebula_cpu_mt    # same
```

> If cmake can't find HDF5: `cmake -DHDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/serial ..`

## Step 3: cstool + Patches

```bash
cd nebula/cstool-master
unzip -o cstool-master.zip
```

Apply 6 patches for ELSEPA 2020 compatibility (all in `cstool-master/`):

### 3a. `setup.py` — setuptools
```bash
sed -i 's/from distutils.core import setup/from setuptools import setup/' setup.py
```

### 3b. `cstool/mott/run_elsepa.py` — symlink resolve
```bash
sed -i "s|os.path.dirname(shutil.which('elscata'))|os.path.dirname(os.path.realpath(shutil.which('elscata')))|" cstool/mott/run_elsepa.py
```

### 3c. `cstool/mott/run_elsepa.py` — glob pattern
```bash
sed -i "s|'dcs\*.dat'|'dcs_*.dat'|" cstool/mott/run_elsepa.py
```

### 3d. `cstool/mott/elsepa_input.py` — VABSA default
```bash
sed -i 's/VABSA=2.0/VABSA=-1.0/' cstool/mott/elsepa_input.py
```

### 3e. `cstool/mott/mott.py` — enable absorption potential
Replace the `run_elscata_helper` function body:
```python
# Before:
	settings = elscata_settings(energies, Z,
		IHEF=0,
		MCPOL=2,
		MUFFIN=False if Z in no_muffin_Z else True)

# After:
	use_muffin = Z not in no_muffin_Z
	settings = elscata_settings(energies, Z,
		IHEF=0,
		MCPOL=2,
		MABS=1 if use_muffin else 0,
		MUFFIN=use_muffin)
```

Or via sed:
```bash
sed -i '/settings = elscata_settings(energies, Z,/{
N;N;N
s/.*/\tuse_muffin = Z not in no_muffin_Z\n\tsettings = elscata_settings(energies, Z,\n\t\tIHEF=0,\n\t\tMCPOL=2,\n\t\tMABS=1 if use_muffin else 0,\n\t\tMUFFIN=use_muffin)/
}' cstool/mott/mott.py
```

### 3f. `cstool/endf/obtain_endf_files.py` — remove pkg_resources
```bash
sed -i 's/from pkg_resources import resource_string, resource_filename/_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")/' cstool/endf/obtain_endf_files.py
sed -i "s|sources = json.loads(resource_string(__name__, '../data/endf_sources.json').decode(\"utf-8\"))|with open(os.path.join(_data_dir, 'endf_sources.json')) as f:\n\t\tsources = json.load(f)|" cstool/endf/obtain_endf_files.py
sed -i "s|endf_dir = resource_filename(__name__, '../data/endf_data')|endf_dir = os.path.join(_data_dir, 'endf_data')|" cstool/endf/obtain_endf_files.py
```

### 3g. `cstool/common/interpolate.py` — numpy warning fix
```bash
sed -i 's/log_y = np.log(y.magnitude, where=y.magnitude>0)/log_y = np.full_like(y.magnitude, -np.inf)/' cstool/common/interpolate.py
sed -i 's/log_y\[y.magnitude <= 0\] = -np.inf/np.log(y.magnitude, where=y.magnitude>0, out=log_y)/' cstool/common/interpolate.py
```

## Step 4: Virtual Environment + Symlinks

```bash
cd <project_root>    # e.g. ~/ML_exploration/NebulaQ_test
uv venv cstool-env
source cstool-env/bin/activate
uv pip install -e nebula/cstool-master/cstool-master/

# Symlinks
ROOT=$(pwd)
ln -sf "$ROOT/nebula/nebula-master/elsepa-2020/elsepa-2020/elscata"              "$ROOT/cstool-env/bin/elscata"
ln -sf "$ROOT/nebula/nebula-master/Nebula-master/build/bin/nebula_gpu"           "$ROOT/cstool-env/bin/nebula_gpu"
ln -sf "$ROOT/nebula/nebula-master/Nebula-master/build/bin/nebula_cpu_mt"        "$ROOT/cstool-env/bin/nebula_cpu_mt"
ln -sf "$ROOT/nebula/nebula-master/Nebula-master/build/bin/nebula_cpu_edep"      "$ROOT/cstool-env/bin/nebula_cpu_edep"
```

Verify:
```bash
which cstool elscata nebula_gpu
```

## Step 5: Compile Materials

```bash
source cstool-env/bin/activate
cd nebula/cstool-master/cstool-master/data/materials
cstool silicon.yaml     # ~2 min
cstool pmma.yaml        # ~5 min
ls -lh *.mat
```

## Step 6: Run Simulation

```bash
mkdir -p run-simulation/{pri,det,images,tri}
cd run-simulation
# Place .tri geometry in tri/, write sem-pri.py + make_image.py (see docs)
# Then:
source ../cstool-env/bin/activate
nebula_gpu tri/sample.tri pri/sem.pri ../nebula/cstool-master/cstool-master/data/materials/silicon.mat > det/output.det
```

---

## Patch Summary

| # | File | What | Why |
|---|------|------|-----|
| 1 | `CMakeLists.txt` | C++14 → C++17 | CUDA 12+ CUB requires C++17 |
| 2 | `setup.py` | distutils → setuptools | distutils removed in Python 3.12 |
| 3 | `run_elsepa.py` | Add `os.path.realpath()` | Symlink resolves to wrong dir |
| 4 | `run_elsepa.py` | `dcs*.dat` → `dcs_*.dat` | ELSEPA 2020 adds `dcs.dat` (5-col) that breaks parser |
| 5 | `elsepa_input.py` | VABSA 2.0 → -1.0 | ELSEPA 2020 uses -1.0 for auto-calc |
| 6 | `mott.py` | Enable MABS=1 for muffin-tin elements | Use LDA-I absorption potential |
| 7 | `obtain_endf_files.py` | pkg_resources → os.path | pkg_resources deprecated |
| 8 | `interpolate.py` | np.log warning fix | Uninitialized memory with `where=` |
| 9 | `elscata.f` compilation | Single file only | GCC 10+ `-fno-common` causes duplicate symbols |

## Version Requirements

| Tool | Minimum | Tested |
|------|---------|--------|
| g++/gcc | 5 | 13.3.0 |
| cmake | 3.18 | 3.28.3 |
| gfortran | any | 13.3.0 |
| HDF5 | 1.8.13 | 1.10.10 |
| Python | 3.8 | 3.12.12 |
| CUDA | 9.2 (optional) | 13.1 |
