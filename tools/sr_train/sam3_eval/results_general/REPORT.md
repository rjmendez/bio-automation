# SAM3 vs Topaz subject mask -- 94-photo cross-taxon batch (2026-10-23)

Two SAM3 prompts per photo: 'informed' (the photo's own iNat common name -- the realistic production case once a species is known) and 'generic' (a fixed per-group noun phrase, e.g. 'plant'/'mushroom'/'bird' -- the pre-ID triage case).

**Overall**: Topaz degenerate/error 42/94 (45%) | SAM3 hit (informed or generic) 62/94 (66%)

| group | n | Topaz degenerate | SAM3 hit (any) | SAM3 informed avg score | SAM3 generic avg score |
|---|---|---|---|---|---|
| Plantae | 8 | 3/8 | 5/8 | 0.198 | 0.344 |
| Fungi | 8 | 7/8 | 4/8 | 0.190 | 0.476 |
| Insecta | 8 | 2/8 | 7/8 | 0.545 | 0.785 |
| Mammalia | 8 | 5/8 | 6/8 | 0.690 | 0.679 |
| Arachnida | 8 | 2/8 | 5/8 | 0.323 | 0.638 |
| Protozoa | 8 | 5/8 | 0/8 | 0.000 | 0.000 |
| Mollusca | 8 | 4/8 | 7/8 | 0.465 | 0.704 |
| Amphibia | 8 | 3/8 | 8/8 | 0.892 | 0.308 |
| Aves | 8 | 2/8 | 4/8 | 0.461 | 0.468 |
| Reptilia | 8 | 4/8 | 7/8 | 0.724 | 0.000 |
| Animalia | 8 | 5/8 | 4/8 | 0.184 | 0.134 |
| Actinopterygii | 6 | 0/6 | 5/6 | 0.535 | 0.727 |

## Full per-photo table

| iconic | taxon | common | Topaz area | SAM3 informed (prompt->score) | SAM3 generic (prompt->score) |
|---|---|---|---|---|---|
| Actinopterygii | Poecilia reticulata | Guppy | 0.026 | 'guppy'->0.980 | 'fish'->0.965 |
| Actinopterygii | Poecilia reticulata | Guppy | 0.041 | 'guppy'->0.934 | 'fish'->0.945 |
| Actinopterygii | Lepomis | Common Sunfishes | 0.548 | 'common sunfishes'->0.656 | 'fish'->0.828 |
| Actinopterygii | Micropterus nigricans | Largemouth Bass | 0.015 | 'largemouth bass'->0.637 | 'fish'->0.660 |
| Actinopterygii | Actinopterygii | Ray-finned Fishes | 0.000 | 'ray-finned fishes'->0.000 | 'fish'->0.000 |
| Actinopterygii | Betta splendens | Siamese Fighting Fish | 0.616 | 'siamese fighting fish'->0.000 | 'fish'->0.961 |
| Amphibia | Anaxyrus americanus | American Toad | 0.832 | 'american toad'->0.949 | 'frog'->0.883 |
| Amphibia | Notophthalmus viridescens | Eastern Newt | 1.000 | 'eastern newt'->0.945 | 'frog'->0.000 |
| Amphibia | Notophthalmus viridescens | Eastern Newt | 0.285 | 'eastern newt'->0.941 | 'frog'->0.000 |
| Amphibia | Anaxyrus americanus | American Toad | 0.979 | 'american toad'->0.938 | 'frog'->0.945 |
| Amphibia | Notophthalmus viridescens | Eastern Newt | 0.984 | 'eastern newt'->0.934 | 'frog'->0.000 |
| Amphibia | Notophthalmus viridescens | Eastern Newt | 0.806 | 'eastern newt'->0.910 | 'frog'->0.000 |
| Amphibia | Notophthalmus viridescens | Eastern Newt | 0.017 | 'eastern newt'->0.789 | 'frog'->0.633 |
| Amphibia | Notophthalmus viridescens | Eastern Newt | 0.220 | 'eastern newt'->0.730 | 'frog'->0.000 |
| Animalia | Apheloria corrugata | Aromatic Cherry Millipede | 1.000 | 'aromatic cherry millipede'->0.824 | 'animal'->0.000 |
| Animalia | Litylenchus crenatae | Beech Leaf Disease | 0.314 | 'beech leaf disease'->0.645 | 'animal'->0.000 |
| Animalia | Apheloria corrugata | Aromatic Cherry Millipede | 0.995 | 'aromatic cherry millipede'->0.000 | 'animal'->0.523 |
| Animalia | Litylenchus crenatae | Beech Leaf Disease | 0.416 | 'beech leaf disease'->0.000 | 'animal'->0.000 |
| Animalia | Juliformia | Round-backed Millipedes | 0.981 | 'round-backed millipedes'->0.000 | 'animal'->0.000 |
| Animalia | Ptyoiulus impressus | Flocculent Tailed Millipede | 0.999 | 'flocculent tailed millipede'->0.000 | 'animal'->0.000 |
| Animalia | Cylindroiulus | Barrel Millipedes | 0.029 | 'barrel millipedes'->0.000 | 'animal'->0.000 |
| Animalia | Julida | Snake Millipedes | 0.954 | 'snake millipedes'->0.000 | 'animal'->0.551 |
| Arachnida | Phidippus audax | Bold Jumping Spider | 0.528 | 'bold jumping spider'->0.977 | 'spider'->0.953 |
| Arachnida | Araneae | Spiders | 0.165 | 'spiders'->0.695 | 'spider'->0.734 |
| Arachnida | Schizocosa | Brush-legged Spiders | 0.002 | 'brush-legged spiders'->0.586 | 'spider'->0.973 |
| Arachnida | Aculus tetanothrix | Willow Bead Gall Mite | 0.212 | - | - |
| Arachnida | Linyphiidae | Sheetweb and Dwarf Weavers | 0.996 | 'sheetweb and dwarf weavers'->0.000 | 'spider'->0.000 |
| Arachnida | Leiobunum |  | 0.441 | 'leiobunum'->0.000 | 'spider'->0.000 |
| Arachnida | Leucauge venusta | Orchard Orbweaver | 0.136 | 'orchard orbweaver'->0.000 | 'spider'->0.977 |
| Arachnida | Leucauge venusta | Orchard Orbweaver | 0.908 | 'orchard orbweaver'->0.000 | 'spider'->0.832 |
| Aves | Cathartes aura | Turkey Vulture | 0.011 | 'turkey vulture'->0.977 | 'bird'->0.977 |
| Aves | Ardea herodias | Great Blue Heron | 0.015 | 'great blue heron'->0.949 | 'bird'->0.930 |
| Aves | Sialia sialis | Eastern Bluebird | 0.801 | 'eastern bluebird'->0.914 | 'bird'->0.934 |
| Aves | Cathartes aura | Turkey Vulture | 0.034 | 'turkey vulture'->0.852 | 'bird'->0.902 |
| Aves | Turdus migratorius | American Robin | 1.000 | 'american robin'->0.000 | 'bird'->0.000 |
| Aves | Anatidae | Ducks, Geese, and Swans | 0.000 | 'ducks, geese, and swans'->0.000 | 'bird'->0.000 |
| Aves | Hirundo rustica | Barn Swallow | 0.541 | 'barn swallow'->0.000 | 'bird'->0.000 |
| Aves | Meleagris gallopavo | Wild Turkey | 1.000 | 'wild turkey'->0.000 | 'bird'->0.000 |
| Fungi | Agaricales | Common Gilled Mushrooms and Allies | 1.000 | 'common gilled mushrooms and allies'->0.980 | 'mushroom'->0.961 |
| Fungi | Fomitopsis betulina | birch polypore | 0.964 | 'birch polypore'->0.543 | 'mushroom'->0.961 |
| Fungi | Tremella mesenterica | witch's butter | 0.990 | "witch's butter"->0.000 | 'mushroom'->0.000 |
| Fungi | Exidia |  | 0.998 | 'exidia'->0.000 | 'mushroom'->0.000 |
| Fungi | Trichaptum biforme | violet-toothed polypore | 0.881 | 'violet-toothed polypore'->0.000 | 'mushroom'->0.918 |
| Fungi | Tremella |  | 0.122 | 'tremella'->0.000 | 'mushroom'->0.000 |
| Fungi | Chlorociboria |  | 1.000 | 'chlorociboria'->0.000 | 'mushroom'->0.000 |
| Fungi | Leccinum longicurvipes |  | 0.985 | 'leccinum longicurvipes'->0.000 | 'mushroom'->0.969 |
| Insecta | Neotibicen canicularis | Northern Dog-day Cicada | 0.479 | 'northern dog-day cicada'->0.973 | 'insect'->0.980 |
| Insecta | Bombus vagans | Half-black Bumble Bee | 0.993 | 'half-black bumble bee'->0.957 | 'insect'->0.961 |
| Insecta | Euschistus | Brown Stink Bugs | 0.234 | 'brown stink bugs'->0.906 | 'insect'->0.957 |
| Insecta | Tachinidae | Tachinid Flies | 0.834 | 'tachinid flies'->0.828 | 'insect'->0.930 |
| Insecta | Zelus luridus | Pale Green Assassin Bug | 0.578 | 'pale green assassin bug'->0.699 | 'insect'->0.977 |
| Insecta | Diapheromera femorata | Northern Walkingstick | 0.576 | 'northern walkingstick'->0.000 | 'insect'->0.586 |
| Insecta | Eudioctria |  | 0.988 | 'eudioctria'->0.000 | 'insect'->0.891 |
| Insecta | Eurosta solidaginis | Goldenrod Gall Fly | 0.784 | 'goldenrod gall fly'->0.000 | 'insect'->0.000 |
| Mammalia | Canis familiaris | Domestic Dog | 0.986 | 'domestic dog'->0.961 | 'animal'->0.949 |
| Mammalia | Canis familiaris | Domestic Dog | 0.082 | 'domestic dog'->0.953 | 'animal'->0.922 |
| Mammalia | Bos taurus | Domestic Cattle | 0.673 | 'domestic cattle'->0.930 | 'animal'->0.945 |
| Mammalia | Canis familiaris | Domestic Dog | 0.061 | 'domestic dog'->0.922 | 'animal'->0.902 |
| Mammalia | Canis familiaris | Domestic Dog | 1.000 | 'domestic dog'->0.879 | 'animal'->0.848 |
| Mammalia | Canis familiaris | Domestic Dog | 0.901 | 'domestic dog'->0.879 | 'animal'->0.863 |
| Mammalia | Odocoileus virginianus | White-tailed Deer | 1.000 | 'white-tailed deer'->0.000 | 'animal'->0.000 |
| Mammalia | Odocoileus virginianus | White-tailed Deer | 0.999 | 'white-tailed deer'->0.000 | 'animal'->0.000 |
| Mollusca | Arion | Arion Slugs | 0.671 | 'arion slugs'->0.977 | 'snail'->0.855 |
| Mollusca | Arion | Arion Slugs | 0.787 | 'arion slugs'->0.977 | 'snail'->0.887 |
| Mollusca | Arion | Arion Slugs | 0.967 | 'arion slugs'->0.949 | 'snail'->0.793 |
| Mollusca | Gastropoda | Gastropods | 0.956 | 'gastropods'->0.816 | 'snail'->0.934 |
| Mollusca | Megapallifera mutabilis | Changeable Mantleslug | 0.586 | 'changeable mantleslug'->0.000 | 'snail'->0.621 |
| Mollusca | Megapallifera mutabilis | Changeable Mantleslug | 0.568 | 'changeable mantleslug'->0.000 | 'snail'->0.641 |
| Mollusca | Pyganodon cataracta | Eastern Floater | 1.000 | 'eastern floater'->0.000 | 'snail'->0.000 |
| Mollusca | Gastrodontidae |  | 1.000 | 'gastrodontidae'->0.000 | 'snail'->0.898 |
| Plantae | Taraxacum | dandelions | 0.714 | 'dandelions'->0.906 | 'plant'->0.000 |
| Plantae | Potentilla indica | mock strawberry | 0.915 | 'mock strawberry'->0.676 | 'plant'->0.547 |
| Plantae | Solanum lycopersicum | tomato | 0.583 | 'tomato'->0.000 | 'plant'->0.000 |
| Plantae | Berberis thunbergii | Japanese barberry | 0.997 | 'japanese barberry'->0.000 | 'plant'->0.738 |
| Plantae | Salix babylonica | weeping willow | 0.028 | 'weeping willow'->0.000 | 'plant'->0.855 |
| Plantae | Elaeagnus umbellata | autumn olive | 0.966 | 'autumn olive'->0.000 | 'plant'->0.000 |
| Plantae | Vaccinium | blueberries, cranberries, and allies | 0.058 | 'blueberries, cranberries, and allies'->0.000 | 'plant'->0.609 |
| Plantae | Celastrus orbiculatus | Asian bittersweet | 0.044 | 'asian bittersweet'->0.000 | 'plant'->0.000 |
| Protozoa | Lycogala | Wolfs Milk | 0.528 | 'wolfs milk'->0.000 | 'organism'->0.000 |
| Protozoa | Metatrichia vesparia | Wasp's Nest Slime Mold | 0.594 | "wasp's nest slime mold"->0.000 | 'organism'->0.000 |
| Protozoa | Lycogala | Wolfs Milk | 0.227 | 'wolfs milk'->0.000 | 'organism'->0.000 |
| Protozoa | Lycogala | Wolfs Milk | 1.000 | 'wolfs milk'->0.000 | 'organism'->0.000 |
| Protozoa | Lycogala | Wolfs Milk | 0.961 | 'wolfs milk'->0.000 | 'organism'->0.000 |
| Protozoa | Ceratiomyxa fruticulosa | Common Coral Slime | 1.000 | 'common coral slime'->0.000 | 'organism'->0.000 |
| Protozoa | Lycogala flavofuscum |  | 0.929 | 'lycogala flavofuscum'->0.000 | 'organism'->0.000 |
| Protozoa | Mycetozoa | slime molds | 1.000 | 'slime molds'->0.000 | 'organism'->0.000 |
| Reptilia | Thamnophis sirtalis | Common Garter Snake | 1.000 | 'common garter snake'->0.938 | 'lizard'->0.000 |
| Reptilia | Plestiodon fasciatus | Common Five-lined Skink | 0.360 | 'common five-lined skink'->0.895 | 'lizard'->0.000 |
| Reptilia | Chrysemys picta | Painted Turtle | 0.999 | 'painted turtle'->0.828 | 'lizard'->0.000 |
| Reptilia | Thamnophis sirtalis | Common Garter Snake | 0.783 | 'common garter snake'->0.805 | 'lizard'->0.000 |
| Reptilia | Thamnophis sirtalis | Common Garter Snake | 0.977 | 'common garter snake'->0.785 | 'lizard'->0.000 |
| Reptilia | Terrapene carolina | Common Box Turtle | 1.000 | 'common box turtle'->0.773 | 'lizard'->0.000 |
| Reptilia | Pantherophis alleghaniensis | Central Ratsnake | 0.653 | 'central ratsnake'->0.770 | 'lizard'->0.000 |
| Reptilia | Nerodia sipedon | Common Watersnake | 0.425 | 'common watersnake'->0.000 | 'lizard'->0.000 |