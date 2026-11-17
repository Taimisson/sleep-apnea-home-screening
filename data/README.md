# Data - UCDDB

This project uses the **St Vincent's University Hospital / University College Dublin Sleep Apnea Database (UCDDB)**, published on PhysioNet.

- **Page:** https://physionet.org/content/ucddb/1.0.0/
- **What it is:** overnight polysomnography of **25 adults** (21 men, 4 women) referred to the sleep disorders clinic of St Vincent's University Hospital, Dublin, for suspected sleep-disordered breathing.
- **What we use:** the **SpO₂** channel of the polysomnography and the **respiratory events scored by a sleep technologist** (apneas and hypopneas), plus the AHI of each subject.
- **License:** Open Data Commons Attribution License v1.0 (ODC-BY 1.0).

> ⚠️ The data is **not** stored in this repository (it is about 1.3 GB and `data/raw/` is in `.gitignore`). Each team member downloads it locally.

## How to download

### Browser (Windows, Linux or macOS)

1. Open https://physionet.org/content/ucddb/1.0.0/
2. At the bottom of the page, under **Files**, click **"Download the ZIP file"**.
3. Extract the ZIP (about 760 MB; 1.3 GB uncompressed) **inside `data/raw/`**. The result must be:

```
data/raw/ucddb-1.0.0/files/
├── RECORDS
├── SHA256SUMS.txt
├── SubjectDetails.xls
├── ucddb002.rec
├── ucddb002_lifecard.edf
├── ucddb002_respevt.txt
├── ucddb002_stage.txt
├── ucddb003.rec
└── ... (25 subjects)
```

### Command line (Linux / macOS)

From the repository root:

```bash
mkdir -p data/raw
curl -L -o data/raw/ucddb.zip https://physionet.org/content/ucddb/get-zip/1.0.0/
unzip data/raw/ucddb.zip -d data/raw/
rm data/raw/ucddb.zip
```

After unzipping, check that the files are in `data/raw/ucddb-1.0.0/files/`. If the ZIP created a folder with a different name, rename or move it to match this path.

The code reads the data from this path (`data/raw/ucddb-1.0.0/files/`). If you extract it somewhere else, the scripts will not find it.

## Check the download

- **25 subjects**, with IDs from 002 to 028 (001, 004 and 016 do not exist): 002, 003, 005, 006, 007, 008, 009, 010, 011, 012, 013, 014, 015, 017, 018, 019, 020, 021, 022, 023, 024, 025, 026, 027, 028.
- Each subject has 4 files (`.rec`, `_lifecard.edf`, `_respevt.txt`, `_stage.txt`), plus `SubjectDetails.xls`, `RECORDS` and `SHA256SUMS.txt` → **103 files** in total.
- Optional integrity check, run **inside `data/raw/ucddb-1.0.0/files/`**:

**Linux** (or Git Bash / WSL on Windows):

```bash
sha256sum -c SHA256SUMS.txt
```

**macOS** (`SHA256SUMS.txt` uses a single space between hash and file name, which `shasum` does not accept directly, so the `sed` adds a second one):

```bash
sed 's/ /  /' SHA256SUMS.txt | shasum -a 256 -c -
```

**Windows** (PowerShell):

```powershell
Get-Content SHA256SUMS.txt | ForEach-Object {
    $expected, $file = $_ -split ' ', 2
    $actual = (Get-FileHash $file -Algorithm SHA256).Hash
    if ($actual -eq $expected) { "${file}: OK" } else { "${file}: FAILED" }
}
```

All 102 lines must end in `OK` (every file except `SHA256SUMS.txt` itself).

- Run `git status` in the repository root: **nothing from `data/raw/` may appear**.

## Files per subject

| File | Content | Used in this project |
|---|---|---|
| `ucddbXXX.rec` | Full polysomnography in EDF format (with a `.rec` extension), including the SpO₂ channel | ✅ SpO₂ signal (and pulse, if useful) |
| `ucddbXXX_respevt.txt` | Respiratory events scored by a sleep technologist (onset time, type, duration, SpO₂ drop, among others) | ✅ Ground truth for event detection |
| `ucddbXXX_stage.txt` | Sleep stage for each 30-second epoch, one value per line | ⚪ Optional (sleep vs. wake time) |
| `ucddbXXX_lifecard.edf` | Simultaneous Holter ECG recording | ❌ Not used |
| `SubjectDetails.xls` | Demographics and clinical summary per subject, including the AHI | ✅ Night-level label (AHI ≥ 15) |

Notes:
- `.rec` files are standard EDF. If a library refuses the `.rec` extension, make a **local copy** with the `.edf` extension (both extensions are ignored by git).
- Event times in `_respevt.txt` are wall-clock times; they must be converted to seconds from the start of the recording (watch out for events after midnight).

## How to cite

The PhysioNet page asks for the database and the standard PhysioNet citation:

- **Database:** St Vincent's University Hospital / University College Dublin Sleep Apnea Database (version 1.0.0). PhysioNet. https://doi.org/10.13026/C26C7D
- **Standard PhysioNet citation:** Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. *Nature Health*, 1(8), 792–795. https://doi.org/10.1038/s44360-026-00096-z

Other citation formats (BibTeX, Vancouver, etc.) are available on the database page.

## Privacy

The recordings are de-identified research data released for public use. Even so, they are health data: they must stay out of the repository, must not be uploaded to public servers (for example, the public HAPI FHIR server), and the patients in the demo are identified only by the UCDDB subject ID (e.g., `ucddb002`).
