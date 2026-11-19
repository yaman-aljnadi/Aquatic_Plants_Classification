# Are the 21 names in TreeOfLife-10M?

Checked on 18 Nov 2026 against the taxonomy files on Hugging Face, not the images. TreeOfLife-10M is about 10 million photos, so the image archive was not downloaded. The match uses `metadata/naming/eol_name_lookup.json` (75.6 MB, 504,018 taxa). That file is the Encyclopedia of Life name table BioCLIP was trained against. A second column uses `metadata/naming/inat21_name_lookup.json` (1.6 MB, the 10,000 iNaturalist 2021 classes). iNaturalist 2021 is one source inside TreeOfLife-10M, and the cleaned web photos in this project are also iNaturalist images, so that column is the overlap that can inflate a web score.

A row is "present at species" when the scientific name is a species node in that file. "Present as genus" means the genus is there and the species epithet is not. "Absent" means the genus is not there either. Cabomba is stored as a genus in `config.py`, so it is matched at genus rank on purpose.

These files list taxon names. They do not say how many photos each name has. A name can be in the taxonomy with few or no training photos.

| Class name | TreeOfLife-10M (EOL names) | iNat21 (10k classes) |
| --- | --- | --- |
| Brasenia schreberi | present at species | present at species |
| Cabomba | present as genus | absent |
| Ceratophyllum demersum | present at species | present at species |
| Elodea canadensis | present at species | present at species |
| Heteranthera dubia | present at species | absent |
| Hydrocharis morsus-ranae | present at species | present at species |
| Myriophyllum sibiricum | present at species | present as genus |
| Myriophyllum spicatum | present at species | present at species |
| Najas flexilis | present at species | absent |
| Nitellopsis obtusa | present at species | absent |
| Nuphar variegata | present at species | present at species |
| Nymphaea odorata | present at species | present at species |
| Potamogeton crispus | present at species | present at species |
| Potamogeton gramineus | present at species | present as genus |
| Potamogeton illinoensis | present at species | present as genus |
| Potamogeton natans | present at species | present at species |
| Potamogeton praelongus | present at species | present as genus |
| Potamogeton richardsonii | present at species | present as genus |
| Potamogeton robbinsii | present at species | present as genus |
| Ranunculus aquatilis | present at species | present at species |
| Vallisneria americana | present at species | absent |

TreeOfLife-10M: 20 names present at species, Cabomba present as genus, none absent. The EOL table does contain Cabomba species (`C. caroliniana`, `C. aquatica`, `C. furcata`, `C. haynesii`, `C. palaeformis`). Our label is the genus, not one of those species.

iNat21: 10 present at species, 6 present only as genus, 5 absent (Cabomba, Heteranthera dubia, Najas flexilis, Nitellopsis obtusa, Vallisneria americana). The web-test leak, if there is one, can only apply to names that were actually in that 10k set.

Sources:

- https://huggingface.co/datasets/imageomics/TreeOfLife-10M/blob/main/metadata/naming/eol_name_lookup.json
- https://huggingface.co/datasets/imageomics/TreeOfLife-10M/blob/main/metadata/naming/inat21_name_lookup.json

Copies used for the search are in `Yaman-Aquatic_Plants_Classification/data/cache/` (gitignored).
