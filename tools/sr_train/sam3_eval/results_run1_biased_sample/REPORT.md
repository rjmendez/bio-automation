# SAM3 vs Topaz subject mask -- 30-photo spider/arachnid batch (2026-10-23)

- Topaz degenerate (mask covers >85% of frame, i.e. failed to isolate subject): **4/30** (13%)
- Topaz hard errors: 0/30
- SAM3 found the subject ("spider" or "arachnid" prompt, confidence>0.5): **23/30** (77%)
- SAM3 hard errors: 0/30
- Both worked: 22  |  SAM3 worked where Topaz failed: **1**  |  Topaz worked, SAM3 missed: 4  |  Both failed: 3

| obs_id | taxon | common | Topaz mask area | SAM3 spider score | SAM3 arachnid score | verdict |
|---|---|---|---|---|---|---|
| 203387372 | Dendryphantina |  | 0.853 | 0.957 | 0.980 | SAM3 wins |
| 233517195 | Aceria |  | 0.881 | 0.000 | 0.000 | Topaz ok, SAM3 missed |
| 204185905 | Demodex folliculorum | Human Follicle Mite | 0.997 | 0.000 | 0.000 | Topaz ok, SAM3 missed |
| 280792591 | Eriophyes cerasicrumena | Black Cherry Leaf Gall Mite | 0.886 | 0.000 | 0.000 | Topaz ok, SAM3 missed |
| 220674448 | Coriarachnini |  | 0.013 | 0.988 | 0.984 | both ok |
| 229276510 | Anyphaeninae |  | 0.405 | 0.980 | 0.977 | both ok |
| 243909283 | Araneus pegnia | Butterfly Orbweaver | 0.647 | 0.980 | 0.977 | both ok |
| 241575571 | Araneus thaddeus | Lattice Orbweaver | 0.326 | 0.980 | 0.977 | both ok |
| 217411291 | Araniella displicata | Six-spotted Orbweaver | 0.008 | 0.980 | 0.980 | both ok |
| 230586672 | Elaver excepta | White sac spider | 0.027 | 0.980 | 0.984 | both ok |
| 280792582 | Araneoidea | Orbweavers and Allies | 0.036 | 0.977 | 0.965 | both ok |
| 242775568 | Argiope aurantia | Yellow Garden Spider | 0.052 | 0.973 | 0.969 | both ok |
| 281385158 | Bassaniana versicolor | Multicolored Bark Crab Spider | 0.011 | 0.969 | 0.969 | both ok |
| 209181494 | Eris |  | 0.630 | 0.969 | 0.973 | both ok |
| 273568116 | Eris militaris | Bronze Jumping Spider | 0.052 | 0.965 | 0.965 | both ok |
| 284698488 | Anyphaena |  | 0.353 | 0.961 | 0.961 | both ok |
| 251361792 | Dolomedes albineus | White-banded Fishing Spider | 0.789 | 0.961 | 0.957 | both ok |
| 281656204 | Eustala |  | 0.322 | 0.957 | 0.961 | both ok |
| 216751096 | Dolomedes tenebrosus | Dark Fishing Spider | 0.080 | 0.953 | 0.949 | both ok |
| 222040190 | Florinda coccinea | Black-tailed Red Sheetweaver | 0.011 | 0.953 | 0.938 | both ok |
| 231205368 | Callidosomatinae |  | 0.730 | 0.949 | 0.969 | both ok |
| 244997472 | Araneus marmoreus | Marbled Orbweaver | 0.457 | 0.945 | 0.887 | both ok |
| 255800260 | Araneidae | Orbweavers | 0.347 | 0.941 | 0.957 | both ok |
| 295362004 | Araneinae | Typical Orbweavers | 0.117 | 0.941 | 0.953 | both ok |
| 273568221 | Cyclosa conica | Conical Trashline Orbweaver | 0.033 | 0.930 | 0.938 | both ok |
| 248910528 | Aculus tetanothrix | Willow Bead Gall Mite | 0.323 | 0.000 | 0.000 | Topaz ok, SAM3 missed |
| 273567961 | Araneae | Spiders | 0.631 | 0.000 | 0.000 | Topaz ok, SAM3 missed |
| 222732347 | Dermacentor variabilis | American Dog Tick | 0.020 | 0.000 | 0.816 | both ok |
| 393085745 | Dictynidae | Meshweavers | 0.048 | 0.000 | 0.000 | Topaz ok, SAM3 missed |
| 365214986 | Eriophyinae |  | 0.840 | 0.000 | 0.000 | Topaz ok, SAM3 missed |