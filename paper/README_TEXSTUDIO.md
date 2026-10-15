# 📜 How to Compile the Golden 68 NeurIPS Research Paper in TeXstudio

This guide provides step-by-step instructions for opening, configuring, and compiling the **NeurIPS 2026 Research Paper** (`main.tex`) in **TeXstudio** or **Overleaf**.

---

## 📂 Project Files Included

All paper source files are located in the `paper/` directory:

```text
golden68-ai-audit-framework/
└── paper/
    ├── main.tex               # Primary LaTeX manuscript (NeurIPS 2026 format)
    ├── references.bib         # BibTeX bibliography file (40+ citations)
    └── README_TEXSTUDIO.md    # This guide
```

---

## 🖥️ Option 1: Compiling in TeXstudio (Local Windows / Linux / macOS)

### Step 1: Open the Project in TeXstudio
1. Launch **TeXstudio**.
2. Go to **File -> Open...** (or press `Ctrl + O`).
3. Navigate to `golden68-ai-audit-framework/paper/`.
4. Select `main.tex` and click **Open**.

### Step 2: Configure Compiler Settings
1. Go to **Options -> Configure TeXstudio...**
2. In the left panel, click on **Build**.
3. Ensure the following default settings are selected:
   - **Default Compiler:** `pdflatex` (or `XeLaTeX`)
   - **Default Bibliography Tool:** `BibTeX`
   - **PDF Viewer:** `Internal PDF Viewer (Embedded)`
4. Click **OK**.

### Step 3: Compile the Document
To compile the document with all TikZ diagrams, mathematical formulas, and bibliography references cleanly resolved:

1. Click **Build & View** (Green double-play button `F6` or press `F5`).
2. Alternatively, run the standard 4-pass compilation sequence:
   - Press `F6` (**Compile / PDFLaTeX**)
   - Press `F8` (**Bibliography / BibTeX**)
   - Press `F6` (**PDFLaTeX** again)
   - Press `F6` (**PDFLaTeX** once more to resolve citations)
3. The built PDF will appear in the embedded right-hand preview panel in TeXstudio!

---

## ☁️ Option 2: Compiling in Overleaf

If you prefer using **Overleaf** for online collaboration:

1. Create a new blank project in Overleaf named `Golden68_NeurIPS_Paper`.
2. Upload `main.tex` and `references.bib` into the root directory of the Overleaf project.
3. Select `pdfLaTeX` or `LaTeX` as the compiler in **Menu -> Compiler**.
4. Click **Recompile**.

---

## 🛠️ Required LaTeX Packages (Auto-Installed by MiKTeX / TeX Live)

The document uses standard, widely supported packages:
- `neurips_2026` (included inline in `main.tex`)
- `amsmath`, `amssymb` (AMS Mathematics)
- `tikz` with `shapes.geometric`, `arrows.meta`, `positioning`, `shadows`, `fit`, `calc` (System Architecture Diagram)
- `booktabs` (Professional Tables)
- `algorithm`, `algorithmic` (Algorithm Pseudo-code)
- `hyperref` (Hyperlinks & DOIs)
- `subcaption` (Subfigures and tables)

If your local LaTeX distribution prompts you to install missing packages on first build, click **Install**.

---

## 🏆 Paper Key Highlights for Publication

- **Venue Format:** NeurIPS Datasets and Benchmarks Track (2026)
- **Visuals:** Native TikZ architecture flowchart included in Section 1 / Figure 1.
- **Formulas:** Formal mathematical formulations for Causality ($\mathcal{C}_{\text{caus}}$), Consistency ($\mathcal{S}_{\text{cons}}$), Compliance ($\mathcal{R}_{\text{comp}}$), and Cohen's Kappa ($\kappa$).
- **Tables:** Comprehensive experimental benchmarks across 5 model scales (GPT-4o, Gemini 2.0 Flash, Llama-3.1-70B, Mixtral-8x7B, and GPT-OSS-120B).
