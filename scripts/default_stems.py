# default_stems.py
"""
Manual stem patterns for Turkish political discourse preprocessing.
Applied AFTER Zemberek lemmatization. 
Categorization mirrors signifier_dictionary.json structure.
"""

# =============================================================================
# POLITICAL FIGURES (proper noun normalization)
# =============================================================================
MANUAL_STEMS_POLITICAL_FIGURES = {
    # Erdoğan and family
    r"recep tayyip erdoğan\w*":         "erdoğan",
    r"recep tayyip\w*":                 "erdoğan",
    r"tayyip erdoğan\w*":               "erdoğan",
    r"tayyip bey\w*":                   "erdoğan",
    r"tayyip\w*":                       "erdoğan",
    r"recep\w*":                        "erdoğan",
    
    # CHP leadership
    r"kemal\s+kılıçdaroğlu\w*":         "kılıçdaroğlu",
    r"kılıçdaroğlu\w*":                 "kılıçdaroğlu",
    r"özgür özel\w*":                   "özgür_özel",
    r"ekrem imamoğlu\w*":               "ekrem_imamoğlu",
    r"ekrem başkan\w*":                 "ekrem_imamoğlu",
    r"muharrem ince\w*":                "muharrem_ince",
    r"mansur yavaş\w*":                 "mansur_yavaş",
    r"bülent ecevit\w*":                "bülent_ecevit",
    r"ecevit\w*":                       "bülent_ecevit",
    
    # MHP leadership
    r"\bdevlet bahçeli\w*":             "devlet_bahçeli",
    r"\bdevlet bey\w*":                 "devlet_bahçeli",
    r"sayın bahçeli\w*":                "devlet_bahçeli",
    
    # AKP and judicial figures
    r"akın\s+gürlek\w*":                  "akın_gürlek",
    r"mehmet\s+şimşek\w*":                "mehmet_şimşek",
    r"berat\s+albayrak\w*":             "berat_albayrak",
    
    # Atatürk (founding figure)
    r"gazi mustafa kemal atatürk\w*":   "atatürk",
    r"mustafa kemal atatürk\w*":        "atatürk",
    r"mustafa kemal\w*":                "atatürk",
    r"gazi\s+atatürk\w*":               "atatürk",
    r"atatürk\w*":                      "atatürk",
    
    # International figures
    r"binyamin\s+netanyahu\w*":         "netanyahu",
    r"benyamin\s+netanyahu\w*":         "netanyahu",
    r"netanyahu\w*":                    "netanyahu",
    r"rahip brunson\w*":                "rahip_brunson",
    r"brunson\w*":                      "rahip_brunson",
}


# =============================================================================
# POLITICAL ORGANIZATIONS (parties, coalitions, institutions)
# =============================================================================
MANUAL_STEMS_POLITICAL_ORGANIZATIONS = {
    # Parties (multi-word first)
    r"cumhuriyet halk parti\w*":        "chp",
    r"milliyetçi hareket parti\w*":     "mhp",
    r"\bak parti\w*":                   "akp",
    r"\bmhp\w*":                        "mhp",
    r"\bchp\w*":                        "chp",
    r"\bakp\w*":                        "akp",
    
    # Coalitions
    r"cumhur ittifakı\w*":              "cumhur_ittifakı",
    r"altılı masa\w*":                  "altılı_masa",
    r"ittifak\w*":                      "ittifak",
    
    # State institutions
    r"türkiye\s+büyük\s+millet\s+meclisi\w*": "tbmm",
    r"merkez banka\w*":                 "merkez_bankası",
    r"mecli\w*":                        "tbmm",
}


# =============================================================================
# MASTER SIGNIFIERS
# =============================================================================
MANUAL_STEMS_MASTER_SIGNIFIERS = {
    r"allah\w*":                        "allah",
    r"ahlak\w*":                        "ahlak",
    r"ahlâk\w*":                        "ahlak",
    r"kardeş\w*":                       "kardeş",
    r"arkadaş\w*":                      "arkadaş",
    r"vicdan\w*":                       "vicdan",
}


# =============================================================================
# EMPTY SIGNIFIER CANDIDATES
# =============================================================================
MANUAL_STEMS_EMPTY_SIGNIFIERS = {
    # Multi-word first
    r"türkiye\s+cumhuriyet\w*":         "türkiye",
    r"türkiye\s+devlet\w*":             "türkiye",
    r"milli irade\w*":                  "milli_irade",
    r"genel başkan\w*":                 "genel_başkan",
    
    # Single-word
    r"\bmillet(?!vekili|lerarası)\w*":  "millet",
    r"milletvekili\w*":                 "milletvekili",
    r"milletlerarası\w*":               "milletlerarası",
    r"halk\w*":                         "halk",
    r"vatan(?!daş)\w*":                 "vatan",
    r"kurtuluş\w*":                     "kurtuluş",
    r"umut\w*":                         "umut",
    r"umud\w*":                         "umut",
    r"ümit\w*":                         "umut",
    r"\bsevgi(?!li)\w*":                "sevgi",
    r"\bonur\w*":                       "onur",
    r"beraber\w*":                      "beraber",
    r"birlikte\w*":                     "birlikte",
    r"egemen\w*":                       "egemen",
    r"irade\w*":                        "irade",
    r"sandık\w*":                       "sandık",
    r"sandığ\w*":                       "sandık",
    r"cumhuriyet\w*":                   "cumhuriyet",
    r"türki\w*":                        "türkiye",
    r"bahar\w*":                        "bahar",
    r"\bdevlet\b":                      "devlet",
    r"demokra\w*":                      "demokrasi",
}


# =============================================================================
# ANTAGONISTIC SIGNIFIERS
# =============================================================================
MANUAL_STEMS_ANTAGONISTIC = {
    # IBB disambiguators FIRST (prevent false positives with halk\w*)
    r"halk\s+ekmek\w*":                 "halk_ekmek",
    r"halk\s+bank\w*":                  "halk_bank",
    
    # Multi-word antagonists
    r"tek adam rejim\w*":               "tek_adam_rejim",
    r"tek adam\w*":                     "tek_adam",
    r"beşli çete\w*":                   "beşli_çete",
    r"havuz medya\w*":                  "havuz_medya",
    r"dış güç\w*":                      "dış_güç",
    r"avrupa\s+birliği\w*":             "avrupa_birliği",
    
    # Political enemies
    r"saray\w*":                        "saray",
    r"iktidar\w*":                      "iktidar",
    r"hükûmet\w*":                      "hükümet",
    r"hükümet\w*":                      "hükümet",
    r"rejim\w*":                        "rejim",
    r"vesayet\w*":                      "vesayet",
    r"cunta\w*":                        "cunta",
    
    # Judicial enemies
    r"mahkeme\w*":                      "mahkeme",
    r"yargı\w*":                        "yargı",
    r"kayyım\w*":                       "kayyım",
    
    # Economic enemies
    r"rant\w*":                         "rant",
    r"israf\w*":                        "israf",
    r"oligark\w*":                      "oligark",
    r"baron\w*":                        "baron",
    r"mafya\w*":                        "mafya",
    r"zümre\w*":                        "zümre",
    r"yandaş\w*":                       "yandaş",
    
    # External
    r"avrupa\w*":                       "avrupa",
    r"avrup\w*":                        "avrupa",
    
    # Religious/political groupings
    r"fetö\w*":                         "fetö",
    r"fetullah\w*":                     "fetö",
    r"cemaat\w*":                       "cemaat",
    
    # Terror discourse
    r"terörist\w*":                     "terörist",
    r"terör(?!ist)\w*":                 "terör",
}


# =============================================================================
# EVENT MARKERS
# =============================================================================
MANUAL_STEMS_EVENT_MARKERS = {
    # Multi-word first
    r"\b15\s+temmuz\b":                 "15_temmuz",
    r"erken seçim\w*":                  "erken_seçim",
    
    # Single-word
    r"seçim\w*":                        "seçim",
    r"darbe\w*":                        "darbe",
    r"kriz\w*":                         "kriz",
    r"buhran\w*":                       "buhran",
    r"savaş\w*":                        "savaş",
    r"covid\w*":                        "covid_pandemi",
    r"pandemi\w*":                      "covid_pandemi",
    r"anayasa\w*":                      "anayasa",
    r"yolsuz\w*":                       "yolsuzluk",
    r"yoksul\w*":                       "yoksulluk",
}


# =============================================================================
# ARTICULATORY GESTURES (rhetorical formulas)
# =============================================================================
MANUAL_STEMS_ARTICULATORY_GESTURES = {
    # Symbolic gestures
    r"kırmızı kart\w*":                 "kırmızı_kart",
    r"sarı kart\w*":                    "sarı_kart",
    r"radikal sevgi\w*":                "radikal_sevgi",
    r"hak hukuk adalet\w*":             "hak_hukuk_adalet",
    
    # Rhetorical fillers and exclamations
    r"allah aşkı\w*":                   "allah_aşkına",
    r"yazık günah\w*":                  "yazık_günah",
    r"falan filan\w*":                  "falan_filan",
    r"kanaat önderi\w*":                "kanaat_önderi",
    r"alın teri\w*":                    "alın_teri",
    r"hodri meydan\w*":                 "hodri_meydan",
    r"hele hele\w*":                    "hele_hele",
    r"tek tek\w*":                      "tek_tek",
    r"tek başına\w*":                   "tek_başına",
    r"kusura bakma\w*":                 "kusura_bakma",
    r"emin olun\w*":                    "emin_olun",  # may not fire due to ol → len 2 filter
    r"açık net\w*":                     "açık_net",
    r"vallaha\s+billahi\w*":            "vallahi_billahi",
    r"yol\s+devam\w*":                  "yol_devam",
    r"sözüm söz\w*":                    "sözüm_söz",
    r"omuz\s+omuz\w*":                  "omuz_omuza",
    r"kul hak\w*":                      "kul_hakkı",
    r"beka sorun\w*":                   "beka_sorunu",
    r"anne baba\w*":                    "anne_baba",
    r"ana baba\w*":                     "anne_baba",
    
    # Positions / political stances
    r"normalleş\w*":                    "normalleşme",
    r"helalleş\w*":                     "helalleşme",
    r"\bhayır\w*":                      "hayır",
}


# =============================================================================
# IDENTITIES
# =============================================================================
MANUAL_STEMS_IDENTITIES = {
    # Class/labor identities
    r"emekli\w*":                       "emekli",
    r"emekçi\w*":                       "emekçi",
    r"işçi\w*":                         "işçi",
    r"memur\w*":                        "memur",
    r"çiftçi\w*":                       "çiftçi",
    r"çiftç\w*":                        "çiftçi",
    r"köylü\w*":                        "köylü",
    r"esnaf\w*":                        "esnaf",
    
    # Demographic identities
    r"genç\w*":                         "genç",
    r"kadın\w*":                        "kadın",
    r"aile\w*":                         "aile",
    r"evlat\w*":                        "evlat",
    r"evlad\w*":                        "evlat",
    
    # Educational
    r"öğrenci\w*":                      "öğrenci",
    r"talebe\w*":                       "öğrenci",
    r"öğretmen\w*":                     "öğretmen",
    
    # National/sacrificial
    r"şehit\w*":                        "şehit",
    r"mehmetçik\w*":                    "mehmetçik",
    r"mazlum\w*":                       "mazlum",
    
    # Ethnic/religious
    r"kürt\w*":                         "kürt",
    r"suriyeli\w*":                     "suriyeli",
    r"suriye(?!li)\w*":                 "suriye",
    
    # Civic
    r"vatandaş\w*":                     "vatandaş",
    r"yurttaş\w*":                      "yurttaş",
    r"seçmen\w*":                       "seçmen",
    r"lider\w*":                        "lider",
}


# =============================================================================
# DEMANDS
# =============================================================================
MANUAL_STEMS_DEMANDS = {
    # Multi-word demands first
    r"asgari ücret\w*":                 "asgari_ücret",
    r"sosyal yardım\w*":                "sosyal_yardım",
    r"sosyal devlet\w*":                "sosyal_devlet",
    r"parlamenter sistem\w*":           "parlamenter_sistem",
    r"sivil toplum\w*":                 "sivil_toplum",
    r"yargı bağımsızlık\w*":            "yargı_bağımsızlık",
    r"hukukun üstünlük\w*":             "hukukun_üstünlük",
    
    # Economic demands
    r"bütçe\w*":                        "bütçe",
    r"maaş\w*":                         "maaş",
    r"döviz\w*":                        "döviz",
    r"lira\w*":                         "lira",
    r"dolar\w*":                        "dolar",
    r"altın\w*":                        "altın",
    r"kuruş\w*":                        "kuruş",
    r"\bzam(?!an)\w*":                  "zam",
    r"konut\w*":                        "konut",
    r"konud\w*":                        "konut",
    
    # Justice & rights
    r"adalet\w*":                       "adalet",
    r"\bhak(?!kari|em|kâri|im)\w*":     "hak",
    r"hukuk\w*":                        "hukuk",
    r"hukuğ\w*":                        "hukuk",
    r"haram\w*":                        "haram",
    
    # Political demands / positions
    r"muhalefet\w*":                    "muhalefet",
    r"bağımsız\w*":                     "bağımsız",
    r"laik\w*":                         "laik",
    r"eşitlik\w*":                      "eşitlik",
    r"özgürlük\w*":                     "özgürlük",
    r"özgürlüğ\w*":                     "özgürlük",
    r"özgür(?!lük|_özel)\w*":           "özgür",
    
    # Services & solutions
    r"hizmet\w*":                       "hizmet",
    r"belediye\w*":                     "belediye",
    r"güvence\w*":                      "güvence",
    r"güvenlik\w*":                     "güvenlik",
    r"güven(?!ce|lik)\w*":              "güven",
    r"çare\w*":                         "çare",
    r"ihtiyaç\w*":                      "ihtiyaç",
    r"ihtiyac\w*":                      "ihtiyaç",
    r"sorun\w*":                        "sorun",
    r"baskı\w*":                        "baskı",
    r"mücadele\w*":                     "mücadele",
    r"talep\w*":                        "talep",
    r"taleb\w*":                        "talep",
    r"\bsöz(?!lü|el|er)\w*":            "söz",
    
    # Historical / temporal
    r"saraçhane\w*":                    "saraçhane",
    r"kuvay\w*":                        "kuvayımilliye",
    r"\bzaman\w*":                      "zaman",
}


# =============================================================================
# FUNCTION WORDS (mostly redundant after Zemberek, kept for safety)
# =============================================================================
MANUAL_STEMS_FUNCTION = {
    r"biz\w*":                          "biz",
    r"\bben(den|i|im|sem|dim|imle|de|le)?\b": "ben",
    r"onlar\w*":                        "onlar",
}


# =============================================================================
# COMBINED DICT
# =============================================================================
DEFAULT_STEMS = {
    **MANUAL_STEMS_POLITICAL_FIGURES,
    **MANUAL_STEMS_POLITICAL_ORGANIZATIONS,
    **MANUAL_STEMS_MASTER_SIGNIFIERS,
    **MANUAL_STEMS_EMPTY_SIGNIFIERS,
    **MANUAL_STEMS_ANTAGONISTIC,
    **MANUAL_STEMS_EVENT_MARKERS,
    **MANUAL_STEMS_ARTICULATORY_GESTURES,
    **MANUAL_STEMS_IDENTITIES,
    **MANUAL_STEMS_DEMANDS,
    **MANUAL_STEMS_FUNCTION,
}