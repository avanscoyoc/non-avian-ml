# Zero-shot label check (training-pool clips only)

AUC = how well the label's score separates the class's training clips from the other classes' training clips. `mapping` labels are used for the N=0 point; `candidates` are shown for review only.

## birdnet (6522 labels)

### device_static — birdnet
- no mapped label (no N=0 point)
- top scored on class clips: Dryobates minor_Lesser Spotted Woodpecker (0.66); Turdus philomelos_Song Thrush (0.68); Erithacus rubecula_European Robin (0.70); Aegolius acadicus_Northern Saw-whet Owl (0.64); Bubo virginianus_Great Horned Owl (0.48); Strix aluco_Tawny Owl (0.60); Emberiza hortulana_Ortolan Bunting (0.62); Strix varia_Barred Owl (0.61); Turdus merula_Eurasian Blackbird (0.64); Zonotrichia leucophrys_White-crowned Sparrow (0.55)
- top separating labels: Arizelocichla masukuensis_Shelley's Greenbul (0.77); Myadestes unicolor_Slate-colored Solitaire (0.77); Phyllomyias nigrocapillus_Black-capped Tyrannulet (0.77); Petroica rodinogaster_Pink Robin (0.77); Phylloscartes poecilotis_Variegated Bristle-Tyrant (0.76); Heliangelus exortis_Tourmaline Sunangel (0.76); Ochthoeca cinnamomeiventris_Slaty-backed Chat-Tyrant (0.76); Regulus madeirensis_Madeira Firecrest (0.75); Swynnertonia swynnertoni_Swynnerton's Robin (0.75); Chlorochrysa phoenicotis_Glistening-green Tanager (0.74)

### generator — birdnet
- no mapped label (no N=0 point)
- top scored on class clips: Engine_Engine (0.80); Power tools_Power tools (0.79); Amblycorypha longinicta_Common Virtuoso Katydid (0.88); Human vocal_Human vocal (0.60); Gastrophryne carolinensis_Eastern Narrow-mouthed Toad (0.86); Siren_Siren (0.73); Dryobates pubescens_Downy Woodpecker (0.68); Emberiza hortulana_Ortolan Bunting (0.64); Archilochus colubris_Ruby-throated Hummingbird (0.84); Melanerpes carolinus_Red-bellied Woodpecker (0.61)
- top separating labels: Amblycorypha longinicta_Common Virtuoso Katydid (0.88); Aplonis atrifusca_Samoan Starling (0.87); Gastrophryne carolinensis_Eastern Narrow-mouthed Toad (0.86); Ducula pacifica_Pacific Imperial-Pigeon (0.86); Sporophila nuttingi_Nicaraguan Seed-Finch (0.85); Archilochus colubris_Ruby-throated Hummingbird (0.84); Psilopogon australis_Little Barbet (0.82); Zosterops mauritianus_Mauritius Gray White-eye (0.82); Neopelma sulphureiventer_Sulphur-bellied Tyrant-Manakin (0.81); Rhipidura javanica_Malaysian Pied-Fantail (0.81)

### traffic — birdnet
- no mapped label (no N=0 point)
- top scored on class clips: Engine_Engine (0.54); Zonotrichia leucophrys_White-crowned Sparrow (0.69); Bubo virginianus_Great Horned Owl (0.63); Human vocal_Human vocal (0.58); Salpinctes obsoletus_Rock Wren (0.60); Anas platyrhynchos_Mallard (0.68); Bubo bubo_Eurasian Eagle-Owl (0.61); Rallus aquaticus_Water Rail (0.62); Gallinula chloropus_Eurasian Moorhen (0.71); Turdus philomelos_Song Thrush (0.68)
- top separating labels: Junco hyemalis_Dark-eyed Junco (0.74); Turdus iliacus_Redwing (0.72); Turdus migratorius_American Robin (0.72); Spinus tristis_American Goldfinch (0.72); Fulica atra_Eurasian Coot (0.72); Gallinula chloropus_Eurasian Moorhen (0.71); Calypte anna_Anna's Hummingbird (0.70); Agelaius phoeniceus_Red-winged Blackbird (0.70); Anas crecca_Green-winged Teal (0.70); Quiscalus quiscula_Common Grackle (0.70)

### engine — birdnet
- mapping ['Engine']: AUC 0.840
- top scored on class clips: Engine_Engine (0.84); Human vocal_Human vocal (0.70); Dog_Dog (0.72); Bubo bubo_Eurasian Eagle-Owl (0.62); Bubo virginianus_Great Horned Owl (0.53); Strix aluco_Tawny Owl (0.63); Asio otus_Long-eared Owl (0.54); Rallus aquaticus_Water Rail (0.62); Porzana porzana_Spotted Crake (0.60); Tyto alba_Barn Owl (0.46)
- top separating labels: Engine_Engine (0.84); Dog_Dog (0.72); Human vocal_Human vocal (0.70); Pseudacris crucifer_Spring Peeper (0.68); Turdus iliacus_Redwing (0.68); Spinus tristis_American Goldfinch (0.66); Scudderia furcata_Fork-tailed Bush Katydid (0.65); Dryobates pubescens_Downy Woodpecker (0.64); Seiurus aurocapilla_Ovenbird (0.64); Siren_Siren (0.64)

### power_tools — birdnet
- mapping ['Power tools']: AUC 0.996
- top scored on class clips: Power tools_Power tools (1.00); Engine_Engine (0.95); Human vocal_Human vocal (0.80); Human non-vocal_Human non-vocal (0.86); Cathartes aura_Turkey Vulture (0.98); Gun_Gun (0.86); Fireworks_Fireworks (0.84); Siren_Siren (0.85); Cygnus olor_Mute Swan (0.82); Apis mellifera_Honey Bee (0.91)
- top separating labels: Power tools_Power tools (1.00); Cathartes aura_Turkey Vulture (0.98); Engine_Engine (0.95); Steatornis caripensis_Oilbird (0.93); Apis mellifera_Honey Bee (0.91); Celeus spectabilis_Rufous-headed Woodpecker (0.89); Chlamydera maculata_Spotted Bowerbird (0.88); Cacatua goffiniana_Tanimbar Corella (0.86); Gun_Gun (0.86); Anthracoceros malayanus_Black Hornbill (0.86)

### human_vocal — birdnet
- mapping ['Human vocal']: AUC 0.965
- top scored on class clips: Human vocal_Human vocal (0.96); Engine_Engine (0.65); Dog_Dog (0.85); Bubo virginianus_Great Horned Owl (0.76); Canis lupus_Gray Wolf (0.76); Tyto alba_Barn Owl (0.63); Haemorhous mexicanus_House Finch (0.86); Strix varia_Barred Owl (0.74); Power tools_Power tools (0.65); Bubo bubo_Eurasian Eagle-Owl (0.67)
- top separating labels: Aplonis panayensis_Asian Glossy Starling (0.97); Human vocal_Human vocal (0.96); Pterorhinus perspicillatus_Masked Laughingthrush (0.96); Aratinga nenday_Nanday Parakeet (0.96); Strix seloputo_Spotted Wood-Owl (0.96); Dinemellia dinemelli_White-headed Buffalo-Weaver (0.94); Periparus venustulus_Yellow-bellied Tit (0.94); Eophona migratoria_Yellow-billed Grosbeak (0.93); Cyanopica cyanus_Azure-winged Magpie (0.93); Anthracoceros albirostris_Oriental Pied-Hornbill (0.93)

### metal_clanging — birdnet
- no mapped label (no N=0 point)
- top scored on class clips: Fireworks_Fireworks (0.91); Engine_Engine (0.54); Power tools_Power tools (0.71); Gun_Gun (0.81); Lithobates catesbeianus_American Bullfrog (0.63); Human vocal_Human vocal (0.57); Human non-vocal_Human non-vocal (0.68); Dryocopus pileatus_Pileated Woodpecker (0.70); Canis lupus_Gray Wolf (0.63); Aegolius acadicus_Northern Saw-whet Owl (0.64)
- top separating labels: Melanerpes superciliaris_West Indian Woodpecker (0.92); Fireworks_Fireworks (0.91); Colaptes fernandinae_Fernandina's Flicker (0.87); Phedina borbonica_Mascarene Martin (0.87); Priotelus temnurus_Cuban Trogon (0.87); Pharomachrus fulgidus_White-tipped Quetzal (0.86); Bolborhynchus orbygnesius_Andean Parakeet (0.86); Ptilotula penicillata_White-plumed Honeyeater (0.86); Pionus sordidus_Red-billed Parrot (0.85); Harpyopsis novaeguineae_New Guinea Eagle (0.85)

### airplane — birdnet
- no mapped label (no N=0 point)
- top scored on class clips: Engine_Engine (0.75); Human vocal_Human vocal (0.74); Dog_Dog (0.74); Branta canadensis_Canada Goose (0.76); Melanerpes carolinus_Red-bellied Woodpecker (0.84); Bubo virginianus_Great Horned Owl (0.53); Anas platyrhynchos_Mallard (0.66); Bubo bubo_Eurasian Eagle-Owl (0.61); Callipepla squamata_Scaled Quail (0.76); Coturnix coturnix_Common Quail (0.66)
- top separating labels: Melanerpes carolinus_Red-bellied Woodpecker (0.84); Aechmophorus occidentalis_Western Grebe (0.78); Elaenia cristata_Plain-crested Elaenia (0.77); Branta canadensis_Canada Goose (0.76); Callipepla squamata_Scaled Quail (0.76); Engine_Engine (0.75); Dog_Dog (0.74); Dryobates pubescens_Downy Woodpecker (0.74); Sciurus carolinensis_Eastern Gray Squirrel (0.74); Melospiza melodia_Song Sparrow (0.74)

### gun — birdnet
- mapping ['Gun']: AUC 0.740
- top scored on class clips: Fireworks_Fireworks (0.90); Picoides arcticus_Black-backed Woodpecker (0.92); Aegolius acadicus_Northern Saw-whet Owl (0.88); Spizella atrogularis_Black-chinned Sparrow (0.87); Picoides tridactylus_Eurasian Three-toed Woodpecker (0.86); Sphyrapicus thyroideus_Williamson's Sapsucker (0.90); Picoides dorsalis_American Three-toed Woodpecker (0.93); Dryocopus pileatus_Pileated Woodpecker (0.92); Campephilus magellanicus_Magellanic Woodpecker (0.98); Asio otus_Long-eared Owl (0.83)
- top separating labels: Leucopternis kuhli_White-browed Hawk (1.00); Pipilo ocai_Collared Towhee (0.99); Parabuteo leucorrhous_White-rumped Hawk (0.99); Menura alberti_Albert's Lyrebird (0.99); Campephilus guatemalensis_Pale-billed Woodpecker (0.99); Bombycilla garrulus_Bohemian Waxwing (0.99); Tetrastes bonasia_Hazel Grouse (0.99); Dicrurus ludwigii_Common Square-tailed Drongo (0.98); Campephilus magellanicus_Magellanic Woodpecker (0.98); Icterus graduacauda_Audubon's Oriole (0.98)

### fireworks — birdnet
- mapping ['Fireworks']: AUC 0.980
- top scored on class clips: Fireworks_Fireworks (0.98); Engine_Engine (0.59); Gun_Gun (0.84); Power tools_Power tools (0.86); Canis latrans_Coyote (0.96); Salpinctes obsoletus_Rock Wren (0.67); Dryocopus martius_Black Woodpecker (0.88); Dryocopus pileatus_Pileated Woodpecker (0.94); Sturnella magna_Eastern Meadowlark (0.67); Dolichonyx oryzivorus_Bobolink (0.65)
- top separating labels: Aegolius funereus_Boreal Owl (1.00); Dryocopus lineatus_Lineated Woodpecker (1.00); Melanerpes superciliaris_West Indian Woodpecker (1.00); Emberiza melanocephala_Black-headed Bunting (1.00); Ficedula hypoleuca_European Pied Flycatcher (1.00); Dendrocopos leucotos_White-backed Woodpecker (1.00); Grus japonensis_Red-crowned Crane (1.00); Schoenicola platyurus_Broad-tailed Grassbird (1.00); Campephilus rubricollis_Red-necked Woodpecker (1.00); Platycercus eximius_Eastern Rosella (0.98)

### human_non_vocal — birdnet
- mapping ['Human non-vocal']: AUC 1.000
- top scored on class clips: Tyto alba_Barn Owl (0.96); Human vocal_Human vocal (1.00); Parabuteo unicinctus_Harris's Hawk (1.00); Myiopsitta monachus_Monk Parakeet (1.00); Canis lupus_Gray Wolf (0.88); Engine_Engine (0.80); Cyanopica cooki_Iberian Magpie (1.00); Glaucidium nana_Austral Pygmy-Owl (1.00); Psilopogon haemacephalus_Coppersmith Barbet (1.00); Human non-vocal_Human non-vocal (1.00)
- top separating labels: Myiotheretes striaticollis_Streak-throated Bush-Tyrant (1.00); Cygnus atratus_Black Swan (1.00); Cyanopica cyanus_Azure-winged Magpie (1.00); Cyanopica cooki_Iberian Magpie (1.00); Biatas nigropectus_White-bearded Antshrike (1.00); Oriolus chlorocephalus_Green-headed Oriole (1.00); Serpophaga subcristata_White-crested Tyrannulet (1.00); Turdus abyssinicus_Abyssinian Thrush (1.00); Oenanthe fusca_Brown Rock Chat (1.00); Serpophaga griseicapilla_Straneck's Tyrannulet (1.00)

### wind — birdnet
- no mapped label (no N=0 point)
- top scored on class clips: Engine_Engine (0.65); Fireworks_Fireworks (0.84); Canis lupus_Gray Wolf (0.72); Sturnella neglecta_Western Meadowlark (0.77); Peucaea cassinii_Cassin's Sparrow (0.71); Centronyx bairdii_Baird's Sparrow (0.80); Gun_Gun (0.75); Eremophila alpestris_Horned Lark (0.78); Calcarius lapponicus_Lapland Longspur (0.81); Calidris virgata_Surfbird (0.87)
- top separating labels: Calidris pygmaea_Spoon-billed Sandpiper (0.93); Tringa incana_Wandering Tattler (0.90); Calidris ruficollis_Red-necked Stint (0.90); Thalasseus bergii_Great Crested Tern (0.89); Cinclodes excelsior_Stout-billed Cinclodes (0.89); Numenius minutus_Little Curlew (0.89); Charadrius modestus_Rufous-chested Dotterel (0.89); Cinclodes albidiventris_Chestnut-winged Cinclodes (0.89); Charadrius wilsonia_Wilson's Plover (0.88); Alaudala raytal_Sand Lark (0.88)

### thunder — birdnet
- no mapped label (no N=0 point)
- top scored on class clips: Engine_Engine (0.66); Peucaea cassinii_Cassin's Sparrow (0.85); Anthus rubescens_American Pipit (0.84); Pluvialis dominica_American Golden-Plover (0.86); Eremophila alpestris_Horned Lark (0.92); Porzana porzana_Spotted Crake (0.82); Numenius phaeopus_Whimbrel (0.93); Anthus chacoensis_Pampas Pipit (0.99); Anthus hellmayri_Hellmayr's Pipit (0.95); Plectrophenax nivalis_Snow Bunting (0.98)
- top separating labels: Pulsatrix koeniswaldiana_Tawny-browed Owl (1.00); Leptasthenura xenothorax_White-browed Tit-Spinetail (1.00); Aratinga jandaya_Jandaya Parakeet (1.00); Tockus kempi_Western Red-billed Hornbill (1.00); Ammomanes deserti_Desert Lark (1.00); Pluvialis fulva_Pacific Golden-Plover (1.00); Pluvialis apricaria_European Golden-Plover (1.00); Buteo brachyurus_Short-tailed Hawk (1.00); Chrysothlypis salmoni_Scarlet-and-white Tanager (1.00); Sylviorthorhynchus yanacensis_Tawny Tit-Spinetail (1.00)

### water — birdnet
- no mapped label (no N=0 point)
- top scored on class clips: Engine_Engine (0.78); Melospiza georgiana_Swamp Sparrow (1.00); Melanerpes carolinus_Red-bellied Woodpecker (1.00); Callipepla californica_California Quail (0.89); Tachycineta bicolor_Tree Swallow (1.00); Dog_Dog (0.89); Setophaga coronata_Yellow-rumped Warbler (0.89); Chaetura pelagica_Chimney Swift (0.97); Junco hyemalis_Dark-eyed Junco (0.89); Spizelloides arborea_American Tree Sparrow (1.00)
- top separating labels: Canachites canadensis_Spruce Grouse (1.00); Tachycineta bicolor_Tree Swallow (1.00); Prunella strophiata_Rufous-breasted Accentor (1.00); Anthracothorax nigricollis_Black-throated Mango (1.00); Tamiasciurus hudsonicus_Red Squirrel (1.00); Tangara cyanoventris_Gilt-edged Tanager (1.00); Yuhina occipitalis_Rufous-vented Yuhina (1.00); Botaurus lentiginosus_American Bittern (1.00); Euphagus cyanocephalus_Brewer's Blackbird (1.00); Numida meleagris_Helmeted Guineafowl (1.00)

## birdnet3 (11560 labels)

### device_static — birdnet3
- no mapped label (no N=0 point)
- top scored on class clips: Pipilo maculatus_Spotted Towhee (0.74); Empidonax difficilis_Western Flycatcher (0.88); Troglodytes aedon_Northern House Wren (0.77); Piranga ludoviciana_Western Tanager (0.84); Catharus guttatus_Hermit Thrush (0.83); Regulus satrapa_Golden-crowned Kinglet (0.83); Vireo gilvus_Warbling Vireo (0.84); Turdus migratorius_American Robin (0.77); Bubo virginianus_Great Horned Owl (0.50); Cardinalis cardinalis_Northern Cardinal (0.69)
- top separating labels: Contopus ochraceus_Ochraceous Pewee (0.90); Sericornis nouhuysi_Large Scrubwren (0.89); Bernieria madagascariensis_Long-billed Bernieria (0.89); Empidonax flavescens_Yellowish Flycatcher (0.89); Lepidocolaptes leucogaster_White-striped Woodcreeper (0.89); Mitrephanes phaeocercus_Northern Tufted Flycatcher (0.89); Mitrephanes olivaceus_Olive Tufted Flycatcher (0.89); Passerella schistacea_Slate-colored Fox Sparrow (0.89); Petroica rodinogaster_Pink Robin (0.89); Sipia nigricauda_Esmeraldas Antbird (0.89)

### generator — birdnet3
- no mapped label (no N=0 point)
- top scored on class clips: Turdus migratorius_American Robin (0.77); Melozone crissalis_California Towhee (0.71); Pipilo maculatus_Spotted Towhee (0.63); Melanerpes formicivorus_Acorn Woodpecker (0.61); Passer domesticus_House Sparrow (0.83); Delphinapterus leucas_Beluga (0.80); Haemorhous mexicanus_House Finch (0.83); Empidonax difficilis_Western Flycatcher (0.78); Cardinalis cardinalis_Northern Cardinal (0.84); Aphelocoma californica_California Scrub Jay (0.68)
- top separating labels: Cecropis badia_Rufous-bellied Swallow (0.93); Pellorneum rostratum_White-chested Babbler (0.92); Iole finschii_Finsch's Bulbul (0.92); Ramphocelus nigrogularis_Masked Crimson Tanager (0.92); Psittacula longicauda_Long-tailed Parakeet (0.92); Lophornis delattrei_Rufous-crested Coquette (0.92); Iole crypta_Buff-vented Bulbul (0.92); Caloramphus fuliginosus_Brown Barbet (0.92); Orthotomus nigriceps_Black-headed Tailorbird (0.92); Arachnothera clarae_Naked-faced Spiderhunter (0.91)

### traffic — birdnet3
- no mapped label (no N=0 point)
- top scored on class clips: Melozone crissalis_California Towhee (0.66); Myiarchus cinerascens_Ash-throated Flycatcher (0.62); Tachycineta thalassina_Violet-green Swallow (0.66); Tyrannus verticalis_Western Kingbird (0.59); Bubo virginianus_Great Horned Owl (0.61); Pipilo maculatus_Spotted Towhee (0.68); Strix aluco_Tawny Owl (0.60); Melanerpes formicivorus_Acorn Woodpecker (0.69); Mimus polyglottos_Northern Mockingbird (0.61); Sturnella neglecta_Western Meadowlark (0.58)
- top separating labels: Tachybaptus ruficollis_Little Grebe (0.77); Calypte anna_Anna's Hummingbird (0.74); Aphelocoma californica_California Scrub Jay (0.74); Psittacara mitratus_Mitred Parakeet (0.73); Sayornis saya_Say's Phoebe (0.73); Glaucidium californicum_Northern Pygmy Owl (0.73); Hemitriccus nidipendulus_Hangnest Tody-Tyrant (0.73); Selasphorus sasin_Allen's Hummingbird (0.72); Lanius borealis_Northern Shrike (0.72); Herpsilochmus atricapillus_Black-capped Antwren (0.72)

### engine — birdnet3
- no mapped label (no N=0 point)
- top scored on class clips: Eremophila alpestris_Horned Lark (0.67); Melozone crissalis_California Towhee (0.62); Bubo virginianus_Great Horned Owl (0.68); Pipilo maculatus_Spotted Towhee (0.56); Callipepla californica_California Quail (0.54); Thryomanes bewickii_Bewick's Wren (0.52); Gryllus campestris_European Field Cricket (0.50); Strix aluco_Tawny Owl (0.66); Hyliola regilla_Pacific Chorus Frog (0.48); Myiarchus cinerascens_Ash-throated Flycatcher (0.58)
- top separating labels: Artemisiospiza belli_Bell's Sparrow (0.72); Peucaea carpalis_Rufous-winged Sparrow (0.72); Emberiza hortulana_Ortolan Bunting (0.72); Artemisiospiza nevadensis_Sagebrush Sparrow (0.71); Colaptes chrysoides_Gilded Flicker (0.71); Strix hadorami_Desert Owl (0.71); Pluvialis apricaria_European Golden Plover (0.70); Curruca nana_Asian Desert Warbler (0.70); Clamator glandarius_Great Spotted Cuckoo (0.70); Bucanetes githagineus_Trumpeter Finch (0.70)

### power_tools — birdnet3
- no mapped label (no N=0 point)
- top scored on class clips: Megaptera novaeangliae_Humpback Whale (0.94); Delphinapterus leucas_Beluga (0.92); Balaena mysticetus_Bowhead Whale (0.89); Orcinus orca_Orca (0.96); Zalophus californianus_California Sea Lion (0.95); Bos taurus (Domestic type)_Cattle (Domestic type) (0.92); Cervus elaphus_Red Deer (0.85); Pagophilus groenlandicus_Harp Seal (0.91); Cardinalis cardinalis_Northern Cardinal (0.71); Leptonychotes weddellii_Weddell Seal (0.82)
- top separating labels: Mycteria leucocephala_Painted Stork (0.97); Gracula religiosa_Common Hill Myna (0.97); Anastomus oscitans_Asian Openbill (0.97); Lutra lutra_Eurasian Otter (0.97); Phoebastria immutabilis_Laysan Albatross (0.97); Homo sapiens_Human (0.97); Pterodroma neglecta_Kermadec Petrel (0.97); Arctocephalus pusillus pusillus_Brown Fur Seal (pusillus) (0.97); Cervus nippon_Sika Deer (0.97); Mandrillus sphinx_Mandrill (0.97)

### human_vocal — birdnet3
- mapping ['Homo Sapiens_Human vocal']: AUC 0.594
- top scored on class clips: Bubo virginianus_Great Horned Owl (0.72); Strix aluco_Tawny Owl (0.72); Charadrius vociferus_Killdeer (0.61); Pyrrhula pyrrhula_Eurasian Bullfinch (0.71); Strix varia_Barred Owl (0.71); Asio otus_Long-eared Owl (0.72); Cardinalis cardinalis_Northern Cardinal (0.65); Loxia curvirostra_Red Crossbill (0.65); Bubo bubo_Eurasian Eagle-Owl (0.75); Turdus migratorius_American Robin (0.60)
- top separating labels: Eophona migratoria_Chinese Grosbeak (0.90); Ictinia mississippiensis_Mississippi Kite (0.90); Sublegatus arenarum_Northern Scrub Flycatcher (0.89); Aviceda leuphotes_Black Baza (0.89); Patagioenas leucocephala_White-crowned Pigeon (0.89); Contopus caribaeus_Cuban Pewee (0.88); Gorsachius melanolophus_Malayan Night Heron (0.88); Butastur indicus_Grey-faced Buzzard (0.88); Chrysocolaptes festivus_White-naped Woodpecker (0.88); Elaenia martinica_Caribbean Elaenia (0.88)

### metal_clanging — birdnet3
- no mapped label (no N=0 point)
- top scored on class clips: Vulpes vulpes_Red Fox (0.84); Colaptes auratus_Northern Flicker (0.74); Turdus migratorius_American Robin (0.59); Bubo virginianus_Great Horned Owl (0.60); Pipilo maculatus_Spotted Towhee (0.59); Aphelocoma californica_California Scrub Jay (0.62); Strix aluco_Tawny Owl (0.67); Cardinalis cardinalis_Northern Cardinal (0.67); Cyanocitta cristata_Blue Jay (0.73); Setophaga virens_Black-throated Green Warbler (0.74)
- top separating labels: Tetrao urogallus_Western Capercaillie (0.94); Columba bollii_Bolle's Pigeon (0.93); Canachites canadensis_Spruce Grouse (0.92); Propithecus edwardsi_Milne-Edwards's Sifaka (0.92); Epimachus fastosus_Black Sicklebill (0.91); Syrmaticus mikado_Mikado Pheasant (0.91); Astur gundlachi_Gundlach's Hawk (0.91); Crossoptilon mantchuricum_Brown Eared Pheasant (0.91); Dendropicos spodocephalus_Eastern Grey Woodpecker (0.91); Alouatta caraya_Black-and-gold Howler Monkey (0.91)

### airplane — birdnet3
- no mapped label (no N=0 point)
- top scored on class clips: Hyliola regilla_Pacific Chorus Frog (0.42); Agelaius phoeniceus_Red-winged Blackbird (0.45); Bubo virginianus_Great Horned Owl (0.77); Callipepla californica_California Quail (0.60); Spinus tristis_American Goldfinch (0.44); Dolichonyx oryzivorus_Bobolink (0.53); Gryllus bimaculatus_Southern Field Cricket (0.66); Cistothorus stellaris_Sedge Wren (0.56); Eumodicogryllus bordigalensis_Verge Cricket (0.71); Tachycineta bicolor_Tree Swallow (0.48)
- top separating labels: Pseudocolopteryx acutipennis_Subtropical Doradito (0.85); Grallaria atuensis_Atuen Antpitta (0.84); Lagonosticta nitidula_Brown Firefinch (0.81); Lessonia rufa_Austral Negrito (0.81); Iridosornis rufivertex_Golden-crowned Tanager (0.81); Phacellodomus ferrugineigula_Orange-breasted Thornbird (0.81); Rhynchotus maculicollis_Huayco Tinamou (0.81); Curruca melanothorax_Cyprus Warbler (0.81); Phylloscopus fuligiventer_Smoky Warbler (0.81); Heliangelus clarisse_Longuemare's Sunangel (0.81)

### gun — birdnet3
- no mapped label (no N=0 point)
- top scored on class clips: Melozone crissalis_California Towhee (0.86); Pipilo maculatus_Spotted Towhee (0.76); Sturnus vulgaris_Common Starling (0.56); Molothrus ater_Brown-headed Cowbird (0.84); Catharus guttatus_Hermit Thrush (0.83); Turdus migratorius_American Robin (0.91); Scolopax minor_American Woodcock (0.68); Cardinalis cardinalis_Northern Cardinal (0.66); Sturnella neglecta_Western Meadowlark (0.63); Corvus corax_Northern Raven (0.82)
- top separating labels: Setophaga coronata_Yellow-rumped Warbler (0.98); Haemorhous purpureus_Purple Finch (0.98); Colaptes auratus_Northern Flicker (0.97); Junco hyemalis_Dark-eyed Junco (0.95); Corvus brachyrhynchos_American Crow (0.94); Menura novaehollandiae_Superb Lyrebird (0.94); Biziura lobata_Musk Duck (0.94); Corthylio calendula_Ruby-crowned Kinglet (0.93); Sciurus griseus_Western Gray Squirrel (0.93); Peltops montanus_Mountain Peltops (0.93)

### fireworks — birdnet3
- no mapped label (no N=0 point)
- top scored on class clips: Hirundo rustica_Barn Swallow (0.86); Bubo virginianus_Great Horned Owl (0.45); Haemorhous mexicanus_House Finch (0.92); Agelaius phoeniceus_Red-winged Blackbird (0.88); Gryllus bimaculatus_Southern Field Cricket (0.53); Myiarchus cinerascens_Ash-throated Flycatcher (0.57); Gryllus campestris_European Field Cricket (0.59); Fringilla coelebs_Eurasian Chaffinch (0.53); Turdus migratorius_American Robin (0.84); Sturnella neglecta_Western Meadowlark (0.76)
- top separating labels: Vanellus leucurus_White-tailed Lapwing (0.98); Lithobates grylio_Pig Frog (0.96); Rallus tenuirostris_Aztec Rail (0.96); Sturnella lilianae_Chihuahuan Meadowlark (0.96); Petrochelidon pyrrhonota_American Cliff Swallow (0.94); Dolichonyx oryzivorus_Bobolink (0.94); Haemorhous mexicanus_House Finch (0.92); Toxostoma curvirostre_Curve-billed Thrasher (0.92); Cincloramphus cruralis_Brown Songlark (0.92); Ploceus nigricollis_Black-necked Weaver (0.92)

### human_non_vocal — birdnet3
- no mapped label (no N=0 point)
- top scored on class clips: Tyto furcata_American Barn Owl (1.00); Tyto javanica_Eastern Barn Owl (1.00); Tyto alba_Western Barn Owl (1.00); Strix varia_Barred Owl (0.68); Bubo virginianus_Great Horned Owl (0.68); Athene noctua_Little Owl (0.76); Cardinalis cardinalis_Northern Cardinal (0.72); Streptopelia decaocto_Eurasian Collared Dove (0.84); Columba palumbus_Common Wood Pigeon (0.64); Ninox boobook_Australian Boobook (0.92)
- top separating labels: Pternistis squamatus_Scaly Spurfowl (1.00); Mentocrex kioloides_Madagascar Forest Rail (1.00); Hypocharmosyna placentis_Red-flanked Lorikeet (1.00); Pternistis harwoodi_Harwood's Spurfowl (1.00); Hypocnemis subflava_Yellow-breasted Warbling Antbird (1.00); Psitteuteles porphyrocephalus_Purple-crowned Lorikeet (1.00); Birrima castanea_Red Tree-ticker (1.00); Psilopogon rubricapillus_Crimson-fronted Barbet (1.00); Hypsipetes affinis_Seram Golden Bulbul (1.00); Pseudophryne raveni_Copper-backed Brood Frog (1.00)

### wind — birdnet3
- no mapped label (no N=0 point)
- top scored on class clips: Hyliola regilla_Pacific Chorus Frog (0.61); Passer domesticus_House Sparrow (0.65); Sturnella neglecta_Western Meadowlark (0.78); Agelaius phoeniceus_Red-winged Blackbird (0.71); Gryllus campestris_European Field Cricket (0.51); Pagophilus groenlandicus_Harp Seal (0.85); Eumodicogryllus bordigalensis_Verge Cricket (0.63); Loxia curvirostra_Red Crossbill (0.76); Cynoscion nebulosus_Seatrout (0.78); Calidris maritima_Purple Sandpiper (0.86)
- top separating labels: Alaudala raytal_Sand Lark (0.92); Xema sabini_Sabine's Gull (0.92); Charadrius melodus_Piping Plover (0.91); Anarhynchus nivosus_Snowy Plover (0.91); Saundersilarus saundersi_Saunders's Gull (0.91); Sterna trudeaui_Snowy-crowned Tern (0.91); Calidris ptilocnemis_Rock Sandpiper (0.90); Sternula antillarum_Least Tern (0.90); Limosa fedoa_Marbled Godwit (0.90); Lanius sphenocercus_Chinese Grey Shrike (0.90)

### thunder — birdnet3
- no mapped label (no N=0 point)
- top scored on class clips: Hyliola regilla_Pacific Chorus Frog (0.70); Bubo virginianus_Great Horned Owl (0.70); Strix aluco_Tawny Owl (0.82); Corvus corax_Northern Raven (0.88); Alauda arvensis_Eurasian Skylark (0.83); Eremophila alpestris_Horned Lark (0.93); Merops apiaster_European Bee-eater (0.87); Loxia curvirostra_Red Crossbill (0.82); Pagophilus groenlandicus_Harp Seal (0.88); Coturnix coturnix_Common Quail (0.93)
- top separating labels: Ammospiza leconteii_LeConte's Sparrow (1.00); Callipepla squamata_Scaled Quail (0.99); Ammospiza nelsoni_Nelson's Sparrow (0.99); Plectorhyncha lanceolata_Striped Honeyeater (0.98); Cistothorus palustris_Marsh Wren (0.98); Laterallus jamaicensis_Black Rail (0.98); Toxostoma crissale_Crissal Thrasher (0.97); Lagopus leucura_White-tailed Ptarmigan (0.96); Perdix perdix_Grey Partridge (0.96); Sialia currucoides_Mountain Bluebird (0.96)

### water — birdnet3
- no mapped label (no N=0 point)
- top scored on class clips: Bubo virginianus_Great Horned Owl (0.86); Cardinalis cardinalis_Northern Cardinal (0.92); Loxia curvirostra_Red Crossbill (0.92); Setophaga coronata_Yellow-rumped Warbler (0.89); Megaptera novaeangliae_Humpback Whale (0.69); Corvus brachyrhynchos_American Crow (0.86); Scolopax minor_American Woodcock (0.94); Turdus migratorius_American Robin (0.69); Passer domesticus_House Sparrow (0.72); Cyanocitta cristata_Blue Jay (0.97)
- top separating labels: Urocyon cinereoargenteus_Gray Fox (1.00); Tigrisoma mexicanum_Bare-throated Tiger Heron (1.00); Poecile atricapillus_Black-capped Chickadee (1.00); Lithobates grylio_Pig Frog (1.00); Lithobates clamitans_Green Frog (1.00); Acris blanchardi_Blanchard's Cricket Frog (1.00); Acris gryllus_Southern Cricket Frog (1.00); Vulpes vulpes_Red Fox (1.00); Sitta europaea_Eurasian Nuthatch (1.00); Archilochus colubris_Ruby-throated Hummingbird (1.00)

## perch2 (14795 labels)

### device_static — perch2
- no mapped label (no N=0 point)
- top scored on class clips: Strix aluco (0.61); Emberiza hortulana (0.67); Turdus merula (0.71); Fringilla coelebs (0.79); Erithacus rubecula (0.78); Strix varia (0.59); Bubo virginianus (0.54); Phylloscopus collybita (0.66); Rallus aquaticus (0.52); Turdus philomelos (0.72)
- top separating labels: Setophaga caerulescens (0.88); Setophaga townsendi (0.84); Setophaga nigrescens (0.82); Setophaga magnolia (0.82); Cardellina canadensis (0.81); Troglodytes pacificus (0.81); Setophaga virens (0.81); Grallaricula nana (0.80); Certhia americana (0.80); Phylloscopus plumbeitarsus (0.79)

### generator — perch2
- no mapped label (no N=0 point)
- top scored on class clips: Engine (0.88); Musical_instrument (0.89); Music (0.89); Vehicle (0.81); Domestic_sounds_and_home_sounds (0.78); Mechanisms (0.84); Motor_vehicle_(road) (0.84); Tools (0.86); Alarm (0.87); Cardinalis cardinalis (0.93)
- top separating labels: Cardinalis cardinalis (0.93); Pterodroma cahow (0.93); Megatibicen tremulus (0.92); Megapomponia intermedia (0.92); Neotibicen tibicen (0.91); Psaltoda harrisii (0.91); Lapposyrphus lapponicus (0.91); Formicarius nigricapillus (0.91); Cinnyris jugularis (0.91); Heliopais personatus (0.91)

### traffic — perch2
- mapping ['Traffic_noise_and_roadway_noise']: AUC 0.381
- top scored on class clips: Strix aluco (0.59); Rallus aquaticus (0.63); Turdus merula (0.71); Phylloscopus collybita (0.81); Gallinula chloropus (0.68); Fringilla coelebs (0.75); Emberiza hortulana (0.63); Tachybaptus ruficollis (0.68); Erithacus rubecula (0.73); Bubo virginianus (0.55)
- top separating labels: Phylloscopus collybita (0.81); Sitta europaea (0.77); Cettia cetti (0.77); Chlorostilbon alice (0.77); Anthus trivialis (0.77); Fringilla coelebs (0.75); Turdus iliacus (0.75); Certhia brachydactyla (0.75); Anthus pratensis (0.74); Fringilla montifringilla (0.74)

### engine — perch2
- mapping ['Engine', 'Idling']: AUC 0.684
- top scored on class clips: Strix aluco (0.67); Emberiza hortulana (0.73); Rallus aquaticus (0.71); Strix varia (0.58); Turdus merula (0.65); Bubo virginianus (0.54); Gallinula chloropus (0.66); Muscicapa striata (0.77); Fringilla coelebs (0.68); Botaurus stellaris (0.68)
- top separating labels: Schismaderma branchi (0.81); Meconema thalassinum (0.80); Poecilimon hamatus (0.80); Chalcophaps indica (0.80); Pedionomus torquatus (0.80); Pachygrapsus crassipes (0.80); Geotrygon montana (0.79); Callorhinus ursinus (0.78); Zentrygon chiriquensis (0.78); Andreiniimon nuptialis (0.78)

### power_tools — perch2
- mapping ['Power_tool', 'Drill']: AUC 0.995
- top scored on class clips: Engine (0.98); Domestic_sounds_and_home_sounds (0.98); Tools (0.99); Vehicle (0.99); Motor_vehicle_(road) (0.98); Mechanisms (0.99); Sawing (0.99); Accelerating_and_revving_and_vroom (0.99); Motorcycle (0.99); Hiss (0.98)
- top separating labels: Power_tool (1.00); Cathartes aura (0.99); Camera (0.99); Sawing (0.99); Drill (0.99); Motorcycle (0.99); Steatornis caripensis (0.99); Tools (0.99); Screech (0.99); Mechanisms (0.99)

### human_vocal — perch2
- mapping ['Speech', 'Human_voice', 'Homo sapiens']: AUC 0.896
- top scored on class clips: Strix varia (0.71); Strix aluco (0.60); Bubo virginianus (0.62); Human_voice (0.89); Asio otus (0.65); Rallus aquaticus (0.54); Turdus merula (0.58); Bubo bubo (0.67); Corvus corax (0.52); Streptopelia decaocto (0.88)
- top separating labels: Conversation (0.97); Homo sapiens (0.97); Iguana iguana (0.96); Loepa oberthuri (0.96); Onychognathus frater (0.96); Chrysopelea paradisi (0.95); Theropithecus gelada (0.95); Pterorhinus berthemyi (0.95); Anabrus cerciata (0.95); Pylodictis olivaris (0.95)

### metal_clanging — perch2
- no mapped label (no N=0 point)
- top scored on class clips: Domestic_sounds_and_home_sounds (0.86); Explosion (0.87); Vehicle (0.84); Door (0.92); Walk_and_footsteps (0.88); Strix aluco (0.53); Thump_and_thud (0.88); Fireworks (0.88); Dendrocopos major (0.78); Rallus aquaticus (0.45)
- top separating labels: Door (0.92); Drawer_open_or_close (0.91); Campephilus guatemalensis (0.90); Pan troglodytes (0.89); Slam (0.89); Skateboard (0.89); Dishes_and_pots_and_pans (0.89); Pongo pygmaeus (0.89); Gorilla gorilla (0.89); Walk_and_footsteps (0.88)

### airplane — perch2
- mapping ['Fixed-wing_aircraft_and_airplane']: AUC 0.477
- top scored on class clips: Strix aluco (0.56); Emberiza hortulana (0.71); Bubo virginianus (0.59); Strix varia (0.55); Rallus aquaticus (0.58); Turdus merula (0.59); Tachybaptus ruficollis (0.72); Fringilla coelebs (0.50); Phylloscopus collybita (0.55); Alauda arvensis (0.60)
- top separating labels: Atlapetes albofrenatus (0.85); Bucco macrodactylus (0.83); Eumodicogryllus theryi (0.83); Nandinia binotata (0.83); Modicogryllus truncatus (0.83); Elimaea parumpunctata (0.83); Cyprinus carpio (0.82); Xerohippus occidentalis (0.82); Euplectes franciscanus (0.81); Myrmotherula multostriata (0.81)

### gun — perch2
- mapping ['Gunshot_and_gunfire']: AUC 0.840
- top scored on class clips: Explosion (0.84); Fireworks (0.87); Gunshot_and_gunfire (0.84); Strix aluco (0.62); Pipilo maculatus (0.78); Dryocopus pileatus (1.00); Asio otus (0.91); Dryocopus martius (0.80); Dendrocopos major (0.90); Corvus corax (0.64)
- top separating labels: Pogonocichla stellata (1.00); Campephilus guatemalensis (1.00); Campephilus haematogaster (1.00); Tragelaphus scriptus (1.00); Crithagra hyposticta (1.00); Colobus guereza (1.00); Pitta reichenowi (1.00); Dishes_and_pots_and_pans (1.00); Tychaedon signata (1.00); Dryocopus pileatus (1.00)

### fireworks — perch2
- mapping ['Fireworks']: AUC 1.000
- top scored on class clips: Explosion (0.92); Hirundo rustica (0.82); Fireworks (1.00); Gunshot_and_gunfire (0.96); Clapping (1.00); Hands (1.00); Haemorhous mexicanus (0.96); Gryllus campestris (0.88); Thump_and_thud (0.96); Dendrocopos major (1.00)
- top separating labels: Tragelaphus scriptus (1.00); Pycnopygius ixoides (1.00); Aphelocoma wollweberi (1.00); Anas bernieri (1.00); Campephilus haematogaster (1.00); Campephilus guatemalensis (1.00); Stigmochelys pardalis (1.00); Campephilus gayaquilensis (1.00); Yungipicus kizuki (1.00); Cercopithecus pogonias (1.00)

### human_non_vocal — perch2
- no mapped label (no N=0 point)
- top scored on class clips: Tyto furcata (1.00); Tyto alba (0.96); Garrulus glandarius (1.00); Cyanocitta stelleri (1.00); Neotibicen robinsonianus (1.00); Cyanopica cooki (1.00); Streptopelia decaocto (0.92); Athene noctua (0.76); Strix varia (0.68); Sturnus vulgaris (0.88)
- top separating labels: Pteroglossus bitorquatus (1.00); Prionodura newtoniana (1.00); Prioniturus waterstradti (1.00); Prioniturus platenae (1.00); Chloephaga poliocephala (1.00); Prioniturus discurus (1.00); Chloephaga rubidiceps (1.00); Hydrobates tethys (1.00); Primolius auricollis (1.00); Presbytis rubicunda (1.00)

### wind — perch2
- mapping ['Wind']: AUC 0.918
- top scored on class clips: Wind (0.92); Explosion (0.83); Thunderstorm (0.91); Thunder (0.91); Vehicle (0.78); Fire (0.83); Water (0.80); Strix aluco (0.41); Fireworks (0.83); Animal (0.69)
- top separating labels: Locustana pardalina (0.94); Wind (0.92); Thunderstorm (0.91); Thunder (0.91); Anarhynchus atrifrons (0.91); Pyxicephalus adspersus (0.90); Anarhynchus peronii (0.90); Xenostrobus neozelanicus (0.90); Vini stepheni (0.89); Pusa hispida (0.88)

### thunder — perch2
- mapping ['Thunder']: AUC 0.994
- top scored on class clips: Thunderstorm (0.99); Thunder (0.99); Wind (0.98); Strix aluco (0.82); Fire (0.91); Water (0.91); Bubo virginianus (0.67); Explosion (0.88); Corvus corax (0.82); Strix varia (0.64)
- top separating labels: Cynanthus forficatus (1.00); Thunder (0.99); Thunderstorm (0.99); Sus domesticus (0.99); Acrocephalus rehsei (0.98); Wind (0.98); Huechys sanguinea (0.98); Stethophyma lineatum (0.97); Train (0.96); Tibicina steveni (0.96)

### water — perch2
- mapping ['Water']: AUC 0.667
- top scored on class clips: Emberiza citrinella (0.94); Erithacus rubecula (0.69); Corvus corax (0.89); Panurus biarmicus (1.00); Emberiza hortulana (0.72); Motacilla alba (0.97); Sylvia atricapilla (0.72); Corvus brachyrhynchos (0.97); Fringilla coelebs (0.78); Domestic_sounds_and_home_sounds (1.00)
- top separating labels: Lophura ignita (1.00); Sphingonotus luteus (1.00); Collocalia dodgei (1.00); Collocalia sumbawae (1.00); Collocalia troglodytes (1.00); Kikihia horologium (1.00); Euscarthmus meloryphus (1.00); Macaca nemestrina (1.00); Eurystomus gularis (1.00); Sporophila schistacea (1.00)

## cnn14 (527 labels)

### device_static — cnn14
- mapping ['Static']: AUC 0.803
- top scored on class clips: Pink noise (0.76); Environmental noise (0.47); White noise (0.73); Insect (0.52); Cricket (0.50); Waterfall (0.78); Vehicle (0.47); Stream (0.70); Rustle (0.72); Outside, rural or natural (0.56)
- top separating labels: Static (0.80); Waterfall (0.78); Pink noise (0.76); Tuning fork (0.75); Noise (0.75); Rain on surface (0.74); Rain (0.74); Raindrop (0.73); White noise (0.73); Rustle (0.72)

### generator — cnn14
- no mapped label (no N=0 point)
- top scored on class clips: Vehicle (0.82); Engine (0.90); Idling (0.90); Medium engine (mid frequency) (0.89); Car (0.82); Motorcycle (0.90); Vibration (0.91); Environmental noise (0.30); Accelerating, revving, vroom (0.88); Outside, rural or natural (0.48)
- top separating labels: Lawn mower (0.92); Sewing machine (0.92); Engine starting (0.91); Vibration (0.91); Chainsaw (0.91); Engine (0.90); Idling (0.90); Motorcycle (0.90); Jackhammer (0.90); Medium engine (mid frequency) (0.89)

### traffic — cnn14
- mapping ['Traffic noise, roadway noise']: AUC 0.720
- top scored on class clips: Vehicle (0.65); Animal (0.61); Environmental noise (0.61); Insect (0.66); Cricket (0.64); Outside, rural or natural (0.66); White noise (0.67); Bird (0.68); Speech (0.47); Rustle (0.67)
- top separating labels: Railroad car, train wagon (0.81); Train (0.80); Field recording (0.80); Rail transport (0.78); Bee, wasp, etc. (0.77); Tick (0.76); Train wheels squealing (0.76); Pigeon, dove (0.75); Fly, housefly (0.74); Ship (0.74)

### engine — cnn14
- mapping ['Engine', 'Idling']: AUC 0.722
- top scored on class clips: Vehicle (0.73); Music (0.59); Train (0.73); Car (0.73); Thunder (0.67); Animal (0.34); Outside, rural or natural (0.52); Aircraft (0.78); Rail transport (0.72); White noise (0.73)
- top separating labels: Gong (0.83); Rumble (0.83); Scary music (0.82); Fixed-wing aircraft, airplane (0.79); Train horn (0.78); Aircraft (0.78); Field recording (0.77); Civil defense siren (0.77); Sine wave (0.77); Soundtrack music (0.77)

### power_tools — cnn14
- mapping ['Power tool', 'Drill']: AUC 0.986
- top scored on class clips: Chainsaw (0.99); Vehicle (0.69); Power tool (0.99); Tools (0.98); Wood (0.97); Motorcycle (0.93); Engine (0.87); Speech (0.50); Sawing (0.99); Car (0.61)
- top separating labels: Chainsaw (0.99); Sawing (0.99); Power tool (0.99); Rub (0.98); Tools (0.98); Wood (0.97); Jackhammer (0.97); Light engine (high frequency) (0.96); Drill (0.95); Angry music (0.95)

### human_vocal — cnn14
- mapping ['Speech']: AUC 0.996
- top scored on class clips: Speech (1.00); Vehicle (0.78); Boat, Water vehicle (0.77); Music (0.68); Animal (0.44); Car (0.79); Outside, urban or manmade (0.87); Wind (0.68); Wind noise (microphone) (0.70); Outside, rural or natural (0.34)
- top separating labels: Speech (1.00); Hubbub, speech noise, speech babble (0.98); Conversation (0.96); Male speech, man speaking (0.96); Chopping (food) (0.95); Narration, monologue (0.95); Inside, public space (0.95); Female speech, woman speaking (0.94); Crowd (0.93); Chatter (0.93)

### metal_clanging — cnn14
- mapping ['Clang']: AUC 0.469
- top scored on class clips: Vehicle (0.56); Door (0.90); Sliding door (0.89); Cupboard open or close (0.90); Animal (0.37); Speech (0.51); Thunder (0.59); Music (0.64); Train (0.69); Inside, small room (0.75)
- top separating labels: Drawer open or close (0.90); Door (0.90); Cupboard open or close (0.90); Sliding door (0.89); Slam (0.89); Dishes, pots, and pans (0.88); Thump, thud (0.85); Crack (0.83); Thunk (0.83); Printer (0.83)

### airplane — cnn14
- mapping ['Fixed-wing aircraft, airplane']: AUC 0.695
- top scored on class clips: Vehicle (0.65); Cricket (0.55); Insect (0.54); Animal (0.54); Aircraft (0.67); White noise (0.80); Fixed-wing aircraft, airplane (0.70); Outside, rural or natural (0.54); Field recording (0.75); Speech (0.34)
- top separating labels: Gong (0.90); Ambient music (0.86); Change ringing (campanology) (0.85); Rumble (0.82); Scary music (0.81); White noise (0.80); Purr (0.79); Microwave oven (0.78); Mechanical fan (0.77); Single-lens reflex camera (0.77)

### gun — cnn14
- mapping ['Gunshot, gunfire']: AUC 1.000
- top scored on class clips: Firecracker (1.00); Fireworks (1.00); Outside, rural or natural (0.58); Environmental noise (0.55); Arrow (1.00); Animal (0.41); Gunshot, gunfire (1.00); Rustle (0.78); Explosion (0.97); Speech (0.53)
- top separating labels: Slam (1.00); Cap gun (1.00); Firecracker (1.00); Bang (1.00); Machine gun (1.00); Fusillade (1.00); Whip (1.00); Fireworks (1.00); Arrow (1.00); Gunshot, gunfire (1.00)

### fireworks — cnn14
- mapping ['Fireworks']: AUC 1.000
- top scored on class clips: Chop (1.00); Clang (1.00); Arrow (1.00); Hammer (1.00); Firecracker (1.00); Fireworks (1.00); Music (0.69); Speech (0.49); Outside, rural or natural (0.29); Cap gun (1.00)
- top separating labels: Door (1.00); Thump, thud (1.00); Arrow (1.00); Chink, clink (1.00); Glass (1.00); Splinter (1.00); Chop (1.00); Whack, thwack (1.00); Skateboard (1.00); Slam (1.00)

### human_non_vocal — cnn14
- no mapped label (no N=0 point)
- top scored on class clips: Music (1.00); Speech (1.00); Animal (0.20); Inside, small room (1.00); Snort (0.96); Vehicle (0.52); Musical instrument (1.00); Snake (0.48); Mouse (0.72); Television (1.00)
- top separating labels: Speech (1.00); Accordion (1.00); Bagpipes (1.00); Didgeridoo (1.00); Scratching (performance technique) (1.00); Pop music (1.00); Hip hop music (1.00); Rock and roll (1.00); Rhythm and blues (1.00); Soul music (1.00)

### wind — cnn14
- mapping ['Wind', 'Wind noise (microphone)']: AUC 0.962
- top scored on class clips: Vehicle (0.81); Wind noise (microphone) (0.96); Wind (0.96); Boat, Water vehicle (0.89); Thunder (0.85); Rustling leaves (0.86); Thunderstorm (0.88); Sailboat, sailing ship (0.93); Animal (0.40); Ship (0.90)
- top separating labels: Wind noise (microphone) (0.96); Wind (0.96); Sailboat, sailing ship (0.93); Eruption (0.92); Ocean (0.92); Waves, surf (0.91); Fire (0.91); Ship (0.90); Boat, Water vehicle (0.89); Thunderstorm (0.88)

### thunder — cnn14
- mapping ['Thunder']: AUC 0.988
- top scored on class clips: Thunder (0.99); Wind (0.92); Wind noise (microphone) (0.92); Thunderstorm (0.99); Vehicle (0.85); Rain (0.96); Boat, Water vehicle (0.89); Rustling leaves (0.91); Ocean (0.92); Waves, surf (0.91)
- top separating labels: Thunder (0.99); Thunderstorm (0.99); Rain (0.96); Rain on surface (0.95); Clickety-clack (0.95); Fire (0.95); Raindrop (0.93); Traffic noise, roadway noise (0.93); Ship (0.93); Bus (0.93)

### water — cnn14
- mapping ['Water']: AUC 1.000
- top scored on class clips: Pour (1.00); Raindrop (1.00); Animal (0.58); Rain (1.00); Thunder (0.44); Vehicle (0.58); Thunderstorm (0.61); Water (1.00); Stream (0.89); Rain on surface (1.00)
- top separating labels: Raindrop (1.00); Liquid (1.00); Slosh (1.00); Squish (1.00); Drip (1.00); Pour (1.00); Trickle, dribble (1.00); Gush (1.00); Fill (with liquid) (1.00); Pump (liquid) (1.00)
