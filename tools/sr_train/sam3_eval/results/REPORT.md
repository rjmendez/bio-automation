# SAM3 vs Topaz subject mask -- 50-photo spider/arachnid batch (2026-10-23)

- Topaz degenerate (mask covers >85% of frame, i.e. failed to isolate subject): **11/50** (22%)
- Topaz hard errors: 0/50
- SAM3 found the subject ("spider" or "arachnid" prompt, confidence>0.5): **42/50** (84%)
- SAM3 hard errors: 0/50
- Both worked: 35  |  SAM3 worked where Topaz failed: **7**  |  Topaz worked, SAM3 missed: 4  |  Both failed: 4

| obs_id | taxon | common | Topaz mask area | SAM3 spider score | SAM3 arachnid score | verdict |
|---|---|---|---|---|---|---|
| 310843729 | Pisaurina mira | American Nursery Web Spider | 0.860 | 0.980 | 0.977 | SAM3 wins |
| 393085719 | Latrodectus variolus | Northern Black Widow | 1.000 | 0.969 | 0.934 | SAM3 wins |
| 213538428 | Xysticus fraternus | Brotherly Ground Crab Spider | 0.856 | 0.938 | 0.926 | SAM3 wins |
| 291632144 | Leucauge venusta | Orchard Orbweaver | 1.000 | 0.926 | 0.914 | SAM3 wins |
| 321966090 | Phalangiidae | Phalangiid Harvestmen | 0.896 | 0.727 | 0.766 | SAM3 wins |
| 281380868 | Frontinella |  | 1.000 | 0.723 | 0.672 | SAM3 wins |
| 255800271 | Pholcus manueli | Manuel's Cellar Spider | 0.992 | 0.688 | 0.000 | SAM3 wins |
| 219130595 | Phalangioidea |  | 0.992 | 0.000 | 0.000 | both failed |
| 311245273 | Phalangium |  | 1.000 | 0.000 | 0.000 | both failed |
| 232800919 | Salticidae | Jumping Spiders | 1.000 | 0.000 | 0.000 | both failed |
| 204185905 | Demodex folliculorum | Human Follicle Mite | 0.997 | 0.000 | 0.000 | both failed |
| 220674448 | Coriarachnini |  | 0.013 | 0.988 | 0.984 | both ok |
| 251473560 | Philodromus rufus |  | 0.632 | 0.984 | 0.984 | both ok |
| 217411291 | Araniella displicata | Six-spotted Orbweaver | 0.008 | 0.980 | 0.980 | both ok |
| 309330668 | Rabidosa |  | 0.228 | 0.980 | 0.980 | both ok |
| 229276510 | Anyphaeninae |  | 0.405 | 0.980 | 0.977 | both ok |
| 243909283 | Araneus pegnia | Butterfly Orbweaver | 0.647 | 0.980 | 0.977 | both ok |
| 241575571 | Araneus thaddeus | Lattice Orbweaver | 0.326 | 0.980 | 0.977 | both ok |
| 237736913 | Zygoballus |  | 0.010 | 0.977 | 0.984 | both ok |
| 216328698 | Maevia |  | 0.437 | 0.973 | 0.973 | both ok |
| 249248205 | Paraphidippus aurantius | emerald jumping spider | 0.018 | 0.973 | 0.977 | both ok |
| 282705431 | Phidippus putnami | Putnam's Jumping Spider | 0.256 | 0.973 | 0.973 | both ok |
| 239032081 | Parasteatoda tepidariorum | Common House Spider | 0.558 | 0.969 | 0.969 | both ok |
| 281385158 | Bassaniana versicolor | Multicolored Bark Crab Spider | 0.011 | 0.969 | 0.969 | both ok |
| 281656233 | Hibana gracilis | garden ghost spider | 0.276 | 0.965 | 0.965 | both ok |
| 273568116 | Eris militaris | Bronze Jumping Spider | 0.052 | 0.965 | 0.965 | both ok |
| 251361792 | Dolomedes albineus | White-banded Fishing Spider | 0.789 | 0.961 | 0.957 | both ok |
| 284698488 | Anyphaena |  | 0.353 | 0.961 | 0.961 | both ok |
| 219130763 | Gea heptagon | Heptagonal Orbweaver | 0.550 | 0.961 | 0.957 | both ok |
| 281656204 | Eustala |  | 0.322 | 0.957 | 0.961 | both ok |
| 273568151 | Neriene radiata | Filmy Dome Spider | 0.000 | 0.957 | 0.895 | both ok |
| 228735721 | Parasteatoda | Colourful Comb-footed Spiders | 0.248 | 0.953 | 0.945 | both ok |
| 284963233 | Hentzia | Long-jawed Jumping Spiders | 0.264 | 0.953 | 0.961 | both ok |
| 365204968 | Frontinella pyramitela | Bowl-and-doily Spider | 0.597 | 0.953 | 0.949 | both ok |
| 216751096 | Dolomedes tenebrosus | Dark Fishing Spider | 0.080 | 0.953 | 0.949 | both ok |
| 231205368 | Callidosomatinae |  | 0.730 | 0.949 | 0.969 | both ok |
| 244997472 | Araneus marmoreus | Marbled Orbweaver | 0.457 | 0.945 | 0.887 | both ok |
| 310843820 | Misumenini |  | 0.276 | 0.945 | 0.922 | both ok |
| 240168388 | Pholcidae | Cellar Spiders | 0.032 | 0.945 | 0.883 | both ok |
| 319749383 | Phidippus audax | Bold Jumping Spider | 0.383 | 0.941 | 0.969 | both ok |
| 255800260 | Araneidae | Orbweavers | 0.347 | 0.941 | 0.957 | both ok |
| 273568221 | Cyclosa conica | Conical Trashline Orbweaver | 0.033 | 0.930 | 0.938 | both ok |
| 241575579 | Pholcus | Common Cellar Spiders | 0.004 | 0.887 | 0.887 | both ok |
| 248796119 | Philodromidae | Running Crab Spiders | 0.586 | 0.883 | 0.863 | both ok |
| 222040188 | Pelegrina proterva |  | 0.620 | 0.770 | 0.723 | both ok |
| 285838078 | Leiobuninae |  | 0.608 | 0.000 | 0.000 | Topaz ok, SAM3 missed |
| 317399471 | Leiobunum vittatum | Banded Harvestman | 0.795 | 0.000 | 0.000 | Topaz ok, SAM3 missed |
| 222732347 | Dermacentor variabilis | American Dog Tick | 0.020 | 0.000 | 0.816 | both ok |
| 273567961 | Araneae | Spiders | 0.631 | 0.000 | 0.000 | Topaz ok, SAM3 missed |
| 291636219 | Leiobunum |  | 0.441 | 0.000 | 0.000 | Topaz ok, SAM3 missed |