# Nebula SEM Simulator — Deploy SOP

Tested: 2026-03-23 / Ubuntu 24.04 / g++ 13.3 / cmake 3.28.3 / CUDA 13.1 / HDF5 1.10.10 / Python 3.12

## Prerequisites

```bash
sudo apt install -y build-essential gfortran cmake git libhdf5-dev python3
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
```

GPU (optional):
```bash
# Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Directory Layout

Place three zip files in此結構：

```
<project_root>/
└── nebula/
    ├── nebula-master/
    │   ├── Nebula-master.zip
    │   └── elsepa-2020/
    │       └── elsepa-2020.zip
    └── cstool-master/
        └── cstool-master.zip
```

---

## Step 1: ELSEPA 2020

```bash
cd nebula/nebula-master/elsepa-2020
unzip -o elsepa-2020.zip && cd elsepa-2020
gfortran -O elscata.f -o elscata
gfortran -O elscatm.f -o elscatm
./elscata < elscata.in   # verify: should produce dcs_*.dat
```

> **GCC 10+ 注意：** 只傳主檔 `elscata.f`，不要加 `radial.f elsepa2020.f`（它已經 INCLUDE 了）。

## Step 2: Nebula

```bash
cd nebula/nebula-master
unzip -o Nebula-master.zip && cd Nebula-master
```

**修改 1** — 開啟 `CMakeLists.txt`（頂層），找到以下兩行，把 `14` 改成 `17`：

```
set(CMAKE_CXX_STANDARD 14)     →  set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)    →  set(CMAKE_CUDA_STANDARD 17)
```

> 原因：CUDA 12+ 的 CUB 要求 C++17。

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
./bin/nebula_gpu       # verify: "This is Nebula version 1.0.2"
```

> 如果 cmake 找不到 HDF5：`cmake -DHDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/serial ..`

## Step 3: cstool

```bash
cd nebula/cstool-master
unzip -o cstool-master.zip && cd cstool-master
```

以下 7 處修改讓 cstool 支援 ELSEPA 2020：

---

**修改 2** — `setup.py`

```
找: from distutils.core import setup
改: from setuptools import setup
```

> 原因：Python 3.12 移除了 distutils。

---

**修改 3** — `cstool/mott/run_elsepa.py`

```
找: elsepa_dir = os.path.dirname(shutil.which('elscata'))
改: elsepa_dir = os.path.dirname(os.path.realpath(shutil.which('elscata')))
```

> 原因：elscata 是 symlink，不加 realpath 會複製到錯誤目錄。

---

**修改 4** — `cstool/mott/run_elsepa.py`（同一個檔案）

```
找: 'dcs*.dat'
改: 'dcs_*.dat'
```

> 原因：ELSEPA 2020 多輸出一個 `dcs.dat`（5 欄），會被 `dcs*.dat` 抓到導致解析失敗。

---

**修改 5** — `cstool/mott/elsepa_input.py`

```
找: MABS=0,       VABSA=2.0,  VABSD=None, IHEF=1
改: MABS=0,       VABSA=-1.0, VABSD=None, IHEF=1
```

> 原因：ELSEPA 2020 用 -1.0 表示自動計算吸收勢強度。

---

**修改 6** — `cstool/mott/mott.py`

找到 `run_elscata_helper` 函式裡的：
```python
	settings = elscata_settings(energies, Z,
		IHEF=0,
		MCPOL=2,
		MUFFIN=False if Z in no_muffin_Z else True)
```

改成：
```python
	use_muffin = Z not in no_muffin_Z
	settings = elscata_settings(energies, Z,
		IHEF=0,
		MCPOL=2,
		MABS=1 if use_muffin else 0,
		MUFFIN=use_muffin)
```

> 原因：啟用 LDA-I 吸收勢。H/N/O 沒有 muffin-tin 半徑，必須 MABS=0。

---

**修改 7** — `cstool/endf/obtain_endf_files.py`

```
找: from pkg_resources import resource_string, resource_filename
改: _data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
```

同一個檔案，再找：
```python
	sources = json.loads(resource_string(__name__, '../data/endf_sources.json').decode("utf-8"))
	endf_dir = resource_filename(__name__, '../data/endf_data')
```
改成：
```python
	with open(os.path.join(_data_dir, 'endf_sources.json')) as f:
		sources = json.load(f)
	endf_dir = os.path.join(_data_dir, 'endf_data')
```

> 原因：pkg_resources 已棄用。

---

**修改 8** — `cstool/common/interpolate.py`

找：
```python
	log_y = np.log(y.magnitude, where=y.magnitude>0)
	log_y[y.magnitude <= 0] = -np.inf
```
改：
```python
	log_y = np.full_like(y.magnitude, -np.inf)
	np.log(y.magnitude, where=y.magnitude>0, out=log_y)
```

> 原因：numpy `where` 不指定 `out` 會產生未初始化記憶體警告。

---

## Step 4: Virtual Environment + Symlinks

```bash
cd <project_root>
uv venv cstool-env
source cstool-env/bin/activate
uv pip install -e nebula/cstool-master/cstool-master/

ROOT=$(pwd)
ln -sf "$ROOT/nebula/nebula-master/elsepa-2020/elsepa-2020/elscata"         "$ROOT/cstool-env/bin/elscata"
ln -sf "$ROOT/nebula/nebula-master/Nebula-master/build/bin/nebula_gpu"      "$ROOT/cstool-env/bin/nebula_gpu"
ln -sf "$ROOT/nebula/nebula-master/Nebula-master/build/bin/nebula_cpu_mt"   "$ROOT/cstool-env/bin/nebula_cpu_mt"
ln -sf "$ROOT/nebula/nebula-master/Nebula-master/build/bin/nebula_cpu_edep" "$ROOT/cstool-env/bin/nebula_cpu_edep"

which cstool elscata nebula_gpu   # verify
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
source ../cstool-env/bin/activate
```

`run-simulation/` 目錄下需要以下檔案：

| File | Purpose |
|------|---------|
| `hourglass_beam.py` | HourglassGaussianBeam class (JMONSEL port) |
| `hourglass-sem-pri.py` | 產生收斂錐形 hourglass beam 入射電子 |
| `make_image.py` | 從 .det 產生 SEM 影像 |
| `tri/pmma.tri` | 幾何定義（材料 0=silicon, 1=pmma） |

```bash
# 1. 產生入射電子
python3 hourglass-sem-pri.py

# 2. 執行模擬
MAT_DIR="../nebula/cstool-master/cstool-master/data/materials"
nebula_gpu tri/pmma.tri pri/hourglass_sem.pri \
    "$MAT_DIR/silicon.mat" "$MAT_DIR/pmma.mat" \
    > det/output.det

# 3. 產生影像
python3 make_image.py
```

> **注意：不要加 `2>&1`。** Nebula info 文字走 stderr，二進制資料走 stdout。混在一起會導致 .det 檔損毀。

`hourglass-sem-pri.py` 預設參數：500 eV、sigma 1 nm、aperture 15 mrad、focus z=30 nm。可依需求調整。

---

## Quick Reference

| # | File | Change | Reason |
|---|------|--------|--------|
| 1 | `CMakeLists.txt` | C++14 → 17 | CUDA 12+ CUB |
| 2 | `setup.py` | distutils → setuptools | Python 3.12 |
| 3 | `run_elsepa.py` | 加 `realpath()` | symlink 路徑錯誤 |
| 4 | `run_elsepa.py` | `dcs*` → `dcs_*` | ELSEPA 2020 多一個 dcs.dat |
| 5 | `elsepa_input.py` | VABSA 2.0 → -1.0 | 2020 版自動計算 |
| 6 | `mott.py` | 加 MABS=1 | 啟用吸收勢 |
| 7 | `obtain_endf_files.py` | 移除 pkg_resources | 已棄用 |
| 8 | `interpolate.py` | np.log out 參數 | 消除警告 |

| Tool | Minimum | Tested |
|------|---------|--------|
| g++/gcc | 5 | 13.3.0 |
| cmake | 3.18 | 3.28.3 |
| gfortran | any | 13.3.0 |
| HDF5 | 1.8.13 | 1.10.10 |
| Python | 3.8 | 3.12.12 |
| CUDA | 9.2 (optional) | 13.1 |
