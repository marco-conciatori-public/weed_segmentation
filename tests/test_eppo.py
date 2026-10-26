import sqlite3

from weedseg.eppo import build_taxa, check_label_rows, rank_of

# (codeid, eppocode, dtcode, parent codeid)
CODES = [
    (1, "1CAFO", "SPT", None),
    (2, "1AMAF", "SPT", 1),
    (3, "1CHES", "SPT", 2),
    (4, "1CHEG", "SPT", 3),
    (5, "CHEAL", "PFL", 4),
]
# (codeid, name, language, preferred)
NAMES = [
    (1, "Caryophyllales", "la", 1), (2, "Amaranthaceae", "la", 1),
    (3, "Chenopodioideae", "la", 1), (4, "Chenopodium", "la", 1),
    (5, "Chenopodium album", "la", 1), (5, "Chenopodium lanceolatum", "la", 0),
    (5, "fat-hen", "en", 0), (5, "farinello comune", "it", 0),
]


def make_export(path):
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE t_codes (codeid INTEGER PRIMARY KEY, eppocode TEXT, dtcode TEXT,
                              status TEXT);
        CREATE TABLE t_names (nameid INTEGER PRIMARY KEY, codeid INTEGER, fullname TEXT,
                              codelang TEXT, preferred INTEGER, status TEXT);
        CREATE TABLE t_links (idlink INTEGER PRIMARY KEY, codeid INTEGER,
                              codeid_parent INTEGER, status TEXT, idlinkcode INTEGER);
    """)
    for codeid, code, datatype, parent in CODES:
        con.execute("INSERT INTO t_codes VALUES (?, ?, ?, 'A')", (codeid, code, datatype))
        if parent is not None:
            con.execute("INSERT INTO t_links (codeid, codeid_parent, status, idlinkcode) "
                        "VALUES (?, ?, 'A', 9)", (codeid, parent))
    for codeid, name, lang, preferred in NAMES:
        con.execute("INSERT INTO t_names (codeid, fullname, codelang, preferred, status) "
                    "VALUES (?, ?, ?, ?, 'A')", (codeid, name, lang, preferred))
    con.commit()
    con.close()
    return path


def test_rank_of():
    assert rank_of("CHEAL", "PFL") == "species"
    assert rank_of("1CHEG", "SPT") == "genus"
    assert rank_of("1AMAF", "SPT") == "family"


def test_build_taxa_walks_the_taxonomy(tmp_path):
    export = make_export(tmp_path / "eppo.sqlite")
    rows = {row["eppo"]: row for row in build_taxa(export, ["CHEAL", "1CHEG", "XXXXX"])}

    species = rows["CHEAL"]
    assert species["rank"] == "species" and species["active"] == "yes"
    assert species["name"] == "Chenopodium album"
    assert species["synonyms"] == "Chenopodium lanceolatum"
    assert (species["genus"], species["family"], species["order"]) == ("1CHEG", "1AMAF", "1CAFO")
    assert species["family_name"] == "Amaranthaceae"
    assert species["common_names_en"] == "fat-hen"
    assert species["common_names_it"] == "farinello comune"
    assert rows["1CHEG"]["genus"] == "1CHEG"  # a genus is its own genus
    assert rows["XXXXX"]["active"] == ""


def test_check_label_rows(tmp_path):
    export = make_export(tmp_path / "eppo.sqlite")
    taxa = {row["eppo"]: row for row in build_taxa(export, ["CHEAL", "XXXXX"])}
    rows = [
        {"source_label": "1", "eppo": "CHEAL", "scientific_name": "Chenopodium album"},
        {"source_label": "2", "eppo": "CHEAL", "scientific_name": "Chenopodium lanceolatum"},
        {"source_label": "3", "eppo": "CHEAL", "scientific_name": "Chenopodium sp."},
        {"source_label": "4", "eppo": "CHEAL", "scientific_name": "Chenopodium agg.",
         "eppo_note": "aggregate"},
        {"source_label": "5", "eppo": "XXXXX", "scientific_name": ""},
        {"source_label": "6", "eppo": "", "scientific_name": ""},
    ]
    problems = check_label_rows(rows, taxa)
    assert len(problems) == 2
    assert problems[0].startswith("label 3:") and problems[1].startswith("label 5:")


def test_check_label_rows_limits(tmp_path):
    export = make_export(tmp_path / "eppo.sqlite")
    taxa = {row["eppo"]: row for row in build_taxa(export, ["CHEAL", "1CAFO", "1CHES"])}
    rows = [
        {"source_label": "1", "eppo": "CHEAL", "scientific_name": "", "one_of": "1CAFO CHEAL",
         "none_of": ""},
        {"source_label": "2", "eppo": "CHEAL", "scientific_name": "", "none_of": "1CHES"},
        {"source_label": "3", "eppo": "CHEAL", "scientific_name": "", "one_of": "YYYYY"},
        {"source_label": "4", "eppo": "", "scientific_name": "", "none_of": "CHEAL"},
    ]
    problems = check_label_rows(rows, taxa)
    assert [p.split(":")[0] for p in problems] == ["label 2", "label 3", "label 4"]
    assert "subfamily" in problems[0]  # the taxa table doesn't record what a subfamily holds
