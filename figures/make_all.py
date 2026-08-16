"""Build every main-text figure and its LaTeX include file.

    python make_all.py            # rebuild figures from the cached matrices
    FORCE=1 python make_all.py    # re-derive the matrices from the raw CSV too
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

SCRIPTS = ["fig1_overview.py", "fig2_structure.py", "fig3_nmf.py",
           "fig4_bilingual.py", "fig5_uspolitics.py"]

CAPTIONS = {
    1: (r"\textbf{The data.} "
        r"\textbf{A}, The 66 societies of World Values Survey wave 7 "
        r"(2017--2023), colored by the language family assigned in this study; "
        r"dots mark states too small to be visible at this scale. "
        r"\textbf{B}, Respondents analyzed per family after the native-born and "
        r"home-language filter, with the number of societies in parentheses. "
        r"\textbf{C}, The 199 value items retained, by thematic category."),
    2: (r"\textbf{Language family tracks the structure of value responses.} "
        r"\textbf{A}, Average-linkage hierarchical clustering of the 66 "
        r"societies on Euclidean distance between mean response profiles; leaf "
        r"labels are colored by language family. "
        r"\textbf{B}, First two principal components of the standardized "
        r"country mean responses. "
        r"\textbf{C}, The nine items at each end of PC1, colored by theme. "
        r"PC1 contrasts traditional and religious responses against secular "
        r"and permissive ones."),
    3: (r"\textbf{Non-negative matrix factorization recovers the same axis.} "
        r"\textbf{A}, Weight on the first of two NMF components, one mark per "
        r"society, grouped by language family; the second component is the "
        r"remainder. "
        r"\textbf{B}, Three-component solution in barycentric coordinates. "
        r"\textbf{C}, The first NMF component against PC1."),
    4: (r"\textbf{Bilingual states split by the language spoken at home.} "
        r"\textbf{A}, Principal components with Canada divided into English- "
        r"(CDE) and French-speaking (CDF) respondents and Kazakhstan into "
        r"Kazakh- (KZK) and Russian-speaking (KZR); the dashed line and number "
        r"give the distance between the two halves of each state. "
        r"\textbf{B,C}, The twelve societies closest to each split population. "
        r"Bars are 95\% delta-method intervals for the distance between two "
        r"sample mean vectors; hollow marks flag societies with fewer than 200 "
        r"respondents."),
    5: (r"\textbf{The United States split by party identification.} "
        r"\textbf{A}, Principal components with the USA divided into "
        r"Democrat- (USD) and Republican-identifying (USR) respondents. "
        r"\textbf{B,C}, The twelve societies closest to each. "
        r"\textbf{D,E}, Rank of each society by closeness to English- and "
        r"French-speaking Canada in the 16--29 and 50-and-over samples. "
        r"Age-split samples are small for several societies, as noted in the "
        r"panel; those rankings should be read as indicative only."),
}


def write_tex():
    for i, caption in CAPTIONS.items():
        (OUT / f"Figure{i}.tex").write_text(
            "\\begin{figure*}[htbp]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=\\linewidth]{{figures/out/Figure{i}.pdf}}\n"
            f"  \\caption{{{caption}}}\n"
            f"  \\label{{fig:figure{i}}}\n"
            "\\end{figure*}\n")
    print(f"wrote {len(CAPTIONS)} LaTeX include files")


def main():
    for script in SCRIPTS:
        print(f"--- {script}")
        r = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE)
        if r.returncode:
            sys.exit(f"{script} failed")
    write_tex()
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:18s} {f.stat().st_size / 1024:8.0f} kB")


if __name__ == "__main__":
    main()
