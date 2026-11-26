# Study 2 — Pt–Pd–Ir–Rh–Ru composition and ORR activity

All experimental data in this directory were digitized from the figures in the publication referred to as Study 2 in the manuscript.

> Banko, L. *et al.* Unravelling Composition–Activity–Stability Trends in High Entropy Alloy Electrocatalysts by Using a Data‐Guided Combinatorial Synthesis Strategy and Computational Modeling. *Adv. Energy Mater.* **12**, 2103312 (2022).

## Composition

`ML1.png` through `ML6.png` show the Pt, Pd, Ir, Rh, and Ru compositions of the six microlibraries (ML1–ML6). `get_compositions_from_png.py` extracts the composition at each grid position and writes one `ML*_composition_data.csv` file per microlibrary. Each row contains the grid coordinates (`Row_Idx`, `Col_Idx`) and the ratio and atom count for each element.

## ORR activity

`PtPdIrRhRu_ORR_ML1-6.png` shows the oxygen reduction reaction (ORR) activity of all six microlibraries. At each composition coordinate, `extract_activity_from_png.py` matches the dot color to the corresponding microlibrary’s colorbar to recover the signed activity, $A$.

The distributed `ML*_activity_data.csv` files contain the activity magnitude scaled linearly within each microlibrary. For $m = |A|$, the saved `Activity` value is $-(m-m_{\min})/(m_{\max}-m_{\min})$, where both extrema are taken over that microlibrary’s 342 positions. The smallest magnitude maps to **0** and the largest to **−1**.

To pair activity with composition, match `Row_Idx` and `Col_Idx` within the same microlibrary.
