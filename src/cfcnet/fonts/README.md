# Figure font

The figures use **TeX Gyre Heros** (a Helvetica-like typeface by B. Jackowski and J. M. Nowacki,
GUST e-foundry), released under the GUST Font License (see `README-TeX-Gyre-Heros.txt`).

The files here are TrueType conversions of the original OpenType (CFF) fonts, made so that
matplotlib embeds them as Type 42 fonts in PDF. As the licence requires for modified versions, the
converted family is renamed **CFCNet Heros**. `src/cfcnet/style.py` registers them at run time, so no
system-wide installation is needed.
