"""The documents of the phase 3 corpus: where each one comes from and which part of it we keep.

``rag/download.py`` reads this list. Every source is downloaded once into ``rag/downloads/``
(out of git) and turned into one or more files under ``rag/corpus/<group>/``, which is what
``rag/ingest.py`` reads.

* **Legal texts** come from Cellar, the Publications Office repository behind EUR-Lex, which
  is meant for automated downloads. EUR-Lex itself answers scripts with a bot check. From each
  agreement we keep the origin chapter or protocol, the introductory notes, the statement on
  origin and the transitional provisions, cut out with the ``start`` and ``end`` patterns of
  each ``Section``. The product-specific rule lists are left out: they are tables, they chunk
  badly, and phase 2 already serves them through ``get_rule``.
* **Guides** are kept whole. The AEAT serves its files to scripts; the Commission's site
  (taxation-customs.ec.europa.eu) answers them with a bot check, so those four are marked
  ``manual``: the script prints where to download them by hand and what to call them.

Licences: Commission and EUR-Lex documents can be reused citing the source (Decision
2011/833/EU); AEAT documents too, under its reuse conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

EU_LICENCE = "Reuse authorised citing the source (Commission Decision 2011/833/EU)"
AEAT_LICENCE = "Reuse authorised citing the source (AEAT reuse conditions)"
CELLAR = "http://publications.europa.eu/resource/cellar/"


@dataclass(frozen=True)
class Section:
    """A part of a legal text that goes into the corpus as a document of its own.

    ``start`` and ``end`` are regular expressions over the extracted text. The section runs
    from the first match of ``start`` to the first match of ``end`` after it (or to the end of
    the text). The patterns include the heading that follows, so the table of contents, where
    headings and titles sit in separate cells, does not match.
    """

    name: str
    title: str
    start: str
    end: str | None


@dataclass(frozen=True)
class Source:
    name: str  # file name in rag/downloads/, without suffix
    title: str
    group: str  # folder under rag/corpus/
    suffix: str  # .xhtml, .html or .pdf
    url: str  # what download.py fetches, or the page to download it from by hand
    reference: str  # how to cite it: CELEX number and EUR-Lex link, or the publisher
    licence: str
    manual: bool = False
    sections: tuple[Section, ...] = field(default=())  # empty: the whole file is kept
    language: str = "en"

    @property
    def file_name(self) -> str:
        return self.name + self.suffix

    def corpus_files(self) -> dict[str, str]:
        """``{path under rag/corpus/: title}`` of the files this source turns into."""
        if not self.sections:
            return {f"{self.group}/{self.file_name}": self.title}
        return {f"{self.group}/{s.name}.md": f"{self.title}: {s.title}" for s in self.sections}


def eurlex(celex: str) -> str:
    return f"CELEX {celex}, https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"


SOURCES: tuple[Source, ...] = (
    # ------------------------------------------------------------------ agreements (Cellar)
    Source(
        name="tca",
        title="EU-UK Trade and Cooperation Agreement",
        group="tca",
        suffix=".xhtml",
        url=CELLAR + "a71c95f6-a99d-11eb-9585-01aa75ed71a1.0006.03/DOC_1",
        reference=eurlex("22021A0430(01)"),
        licence=EU_LICENCE,
        sections=(
            Section(
                "rules_of_origin",
                "Part Two, Heading One, Title I, Chapter 2: Rules of origin",
                r"CHAPTER 2\s+RULES OF ORIGIN\s+SECTION 1",
                r"CHAPTER 3\s+SANITARY AND PHYTOSANITARY",
            ),
            Section(
                "annex2_introductory_notes",
                "Annex 2: Introductory notes to product-specific rules of origin",
                r"ANNEX 2\s+INTRODUCTORY NOTES TO PRODUCT-SPECIFIC RULES OF ORIGIN\s+NOTE 1",
                r"ANNEX 3\s+PRODUCT-SPECIFIC RULES OF ORIGIN\s+Column 1",
            ),
            Section(
                "annex5_electrified_vehicles",
                "Annex 5: Transitional product-specific rules for electric accumulators and "
                "electrified vehicles",
                r"ANNEX 5\s+TRANSITIONAL PRODUCT-SPECIFIC RULES",
                r"ANNEX 6\s+SUPPLIER",
            ),
            Section(
                "annex6_7_supplier_declaration_statement",
                "Annexes 6 and 7: Supplier's declaration and text of the statement on origin",
                r"ANNEX 6\s+SUPPLIER.S DECLARATION\s+1\.",
                r"ANNEX 8\s+JOINT DECLARATION CONCERNING THE PRINCIPALITY OF ANDORRA",
            ),
        ),
    ),
    Source(
        name="japan",
        title="EU-Japan Economic Partnership Agreement",
        group="japan",
        suffix=".xhtml",
        url=CELLAR + "d40c8f20-09a4-11e9-81b4-01aa75ed71a1.0006.02/DOC_1",
        reference=eurlex("22018A1227(01)"),
        licence=EU_LICENCE,
        sections=(
            Section(
                "chapter3_rules_of_origin",
                "Chapter 3: Rules of origin and origin procedures",
                r"CHAPTER 3\s+RULES OF ORIGIN AND ORIGIN PROCEDURES\s+SECTION A\s+Rules of "
                r"origin\s+ARTICLE 3\.1",
                r"CHAPTER 4\s+CUSTOMS MATTERS AND TRADE FACILITATION\s+ARTICLE 4\.1",
            ),
            Section(
                "annex3a_introductory_notes",
                "Annex 3-A: Introductory notes to product specific rules of origin",
                r"ANNEX 3-A\s+INTRODUCTORY NOTES TO PRODUCT SPECIFIC RULES OF ORIGIN\s+Note 1",
                r"ANNEX 3-B\s+PRODUCT SPECIFIC RULES OF ORIGIN\s+Column 1",
            ),
            Section(
                "appendix3b1_vehicles",
                "Appendix 3-B-1: Provisions related to certain vehicles and parts of vehicles",
                r"APPENDIX 3-B-1\s+PROVISIONS RELATED TO CERTAIN VEHICLES AND PARTS OF "
                r"VEHICLES\s+SECTION 1",
                r"ANNEX 3-C\s+INFORMATION REFERRED TO IN ARTICLE 3\.5\s+The information",
            ),
            Section(
                "annex3c_3d_information_statement",
                "Annexes 3-C and 3-D: Information for verification and text of the statement "
                "on origin",
                r"ANNEX 3-C\s+INFORMATION REFERRED TO IN ARTICLE 3\.5\s+The information",
                r"ANNEX 3-E\s+ON THE PRINCIPALITY OF ANDORRA\s+1\.",
            ),
        ),
    ),
    Source(
        name="korea_protocol",
        title="EU-Korea Free Trade Agreement, Protocol concerning the definition of "
        "'originating products' and methods of administrative cooperation",
        group="korea",
        suffix=".html",
        url=CELLAR + "44a9739f-f054-4e7a-bcd9-b3fe3a9bdd05.0007.01/DOC_1",
        reference=eurlex("22011A0514(02)"),
        licence=EU_LICENCE,
        sections=(
            Section(
                "protocol_rules_of_origin",
                "Protocol on origin, Titles I to VIII and Annex I (introductory notes)",
                r"TITLE I\s+General provisions\s+Article 1\s+Definitions",
                r"ANNEX II\s+LIST OF WORKING OR PROCESSING",
            ),
            Section(
                "annex3_origin_declaration",
                "Protocol on origin, Annex III: Text of the origin declaration",
                r"ANNEX III\s+TEXT OF THE ORIGIN DECLARATION",
                r"ANNEX IV\s+COMMITTEE ON OUTWARD PROCESSING ZONES",
            ),
        ),
    ),
    Source(
        name="ceta",
        title="EU-Canada Comprehensive Economic and Trade Agreement (CETA)",
        group="ceta",
        suffix=".xhtml",
        url=CELLAR + "0cf17703-d9e8-11e6-ad7c-01aa75ed71a1.0006.03/DOC_1",
        reference=eurlex("22017A0114(01)"),
        licence=EU_LICENCE,
        sections=(
            Section(
                "protocol_rules_of_origin",
                "Protocol on rules of origin and origin procedures, Sections A to E and "
                "Annexes 1 to 4",
                r"Protocol on rules of origin and origin procedures\s+SECTION A\s+GENERAL "
                r"PROVISIONS",
                r"ANNEX 5\s+PRODUCT-SPECIFIC RULES OF ORIGIN\s+Introductory Notes",
            ),
            Section(
                "annex5_introductory_notes",
                "Protocol on rules of origin, Annex 5: Introductory notes",
                r"ANNEX 5\s+PRODUCT-SPECIFIC RULES OF ORIGIN\s+Introductory Notes",
                r"Harmonized System classification\s+Product specific rule for sufficient "
                r"production",
            ),
            Section(
                "annex6_textiles_declaration",
                "Protocol on rules of origin, Annex 6: Joint declaration on rules of origin "
                "for textiles and apparel",
                r"ANNEX 6\s+JOINT DECLARATION CONCERNING RULES OF ORIGIN FOR TEXTILES",
                r"ANNEX 7\s+JOINT DECLARATIONS CONCERNING THE PRINCIPALITY OF ANDORRA",
            ),
        ),
    ),
    Source(
        name="pem",
        title="Decision No 1/2023 of the PEM Joint Committee amending the Regional Convention "
        "on pan-Euro-Mediterranean preferential rules of origin (revised PEM rules)",
        group="pem",
        suffix=".xhtml",
        url=CELLAR + "2371884d-cecb-11ee-b9d9-01aa75ed71a1.0006.03/DOC_1",
        reference=eurlex("22024D0390"),
        licence=EU_LICENCE,
        sections=(
            Section(
                "appendix1_rules_of_origin",
                "Appendix I, Titles I to VIII and Annex I (introductory notes to the list)",
                r"TITLE I\s+GENERAL PROVISIONS\s+Article 1\s+Definitions\s+For the purposes "
                r"of this Convention",
                r"ANNEX II\s+LIST OF WORKING OR PROCESSING",
            ),
            Section(
                "appendix1_annex3_origin_declaration",
                "Appendix I, Annex III: Text of the origin declaration",
                r"ANNEX III\s+TEXT OF THE ORIGIN DECLARATION",
                r"ANNEX IV\s+SPECIMENS OF MOVEMENT CERTIFICATE",
            ),
            Section(
                "appendix1_annex6_7_supplier_declarations",
                "Appendix I, Annexes VI and VII: Supplier's declaration and long-term "
                "supplier's declaration",
                r"ANNEX VI\s+SUPPLIER.S DECLARATION\s+The supplier",
                r"ANNEX VIII\s+LIST OF CONTRACTING PARTIES",
            ),
        ),
    ),
    Source(
        name="mercosur",
        title="EU-Mercosur Interim Agreement on Trade",
        group="mercosur",
        suffix=".xhtml",
        url=CELLAR + "4c9693e4-1389-11f1-8870-01aa75ed71a1.0006.03/DOC_1",
        reference=eurlex("22026A00184"),
        licence=EU_LICENCE,
        sections=(
            Section(
                "chapter3_rules_of_origin",
                "Chapter 3: Rules of origin and origin procedures",
                r"CHAPTER 3\s+RULES OF ORIGIN AND ORIGIN PROCEDURES\s+SECTION A",
                r"CHAPTER 4\s+CUSTOMS AND TRADE FACILITATION\s+ARTICLE 4\.1",
            ),
            Section(
                "annex3a_introductory_notes",
                "Annex 3-A: Introductory notes to product-specific rules",
                r"ANNEX 3-A\s+INTRODUCTORY NOTES TO PRODUCT-SPECIFIC RULES\s+Note 1",
                r"ANNEX 3-B\s+PRODUCT-SPECIFIC RULES OF ORIGIN\s+Column 1",
            ),
            Section(
                "annex3c_3e_statement_transitional",
                "Annexes 3-C to 3-E: Statement on origin, transitional measures and "
                "administrative errors",
                r"ANNEX 3-C\s+STATEMENT ON ORIGIN\s+The statement",
                r"ANNEX 3-F\s+JOINT DECLARATIONS",
            ),
        ),
    ),
    # ------------------------------------------------- Combined Nomenclature notes (Cellar)
    Source(
        name="cn_explanatory_notes",
        title="Explanatory notes to the Combined Nomenclature of the European Union "
        "(consolidated text of 13.2.2026)",
        group="cn_notes",
        suffix=".xhtml",
        url=CELLAR + "659e8b49-590a-11f1-b3e2-01aa75ed71a1.0006.02/DOC_1",
        reference=eurlex("02019XC0329(02)-20260213"),
        licence=EU_LICENCE,
        sections=(
            Section(
                "section16_chapters84_85",
                "Section XVI, Chapters 84 and 85: Machinery, mechanical appliances and "
                "electrical equipment",
                r"SECTION XVI\s+MACHINERY AND MECHANICAL APPLIANCES",
                r"SECTION XVII\s+VEHICLES, AIRCRAFT",
            ),
            Section(
                "chapter87_vehicles",
                "Chapter 87: Vehicles other than railway or tramway rolling stock",
                r"CHAPTER 87\s+VEHICLES OTHER THAN RAILWAY",
                r"CHAPTER 88\s+AIRCRAFT",
            ),
            Section(
                "chapter94_furniture",
                "Chapter 94: Furniture, bedding, lamps and lighting fittings, prefabricated "
                "buildings",
                r"CHAPTER 94\s+FURNITURE",
                r"CHAPTER 95\s+TOYS",
            ),
        ),
    ),
    # ------------------------------------------------------------ guides served by the AEAT
    Source(
        name="guidance_preferential_origin_2025",
        title="European Commission, Guidance on preferential rules of origin (March 2025)",
        group="guides",
        suffix=".pdf",
        url="https://sede.agenciatributaria.gob.es/static_files/Sede/Tema/Aduanas/"
        "Deuda_aduanera/origen/ayuda/Guidance_March_2025.pdf",
        reference="European Commission (DG TAXUD), hosted by the AEAT",
        licence=EU_LICENCE,
    ),
    Source(
        name="aeat_authorised_exporter",
        title="AEAT, Guía sobre el exportador autorizado",
        group="guides",
        suffix=".pdf",
        url="https://sede.agenciatributaria.gob.es/static_files/AEAT/Aduanas/"
        "Contenidos_Privados/Procedimientos_aduaneros/Origen/guia_export_aut.pdf",
        reference="Agencia Estatal de Administración Tributaria",
        licence=AEAT_LICENCE,
        language="es",
    ),
    Source(
        name="aeat_rex",
        title="AEAT, Guía sobre el sistema de exportadores registrados (REX), mayo de 2022",
        group="guides",
        suffix=".pdf",
        url="https://sede.agenciatributaria.gob.es/static_files/Sede/Tema/Aduanas/"
        "Novedades_aduanas/2025/REX_Guidance_NF.pdf",
        reference="Agencia Estatal de Administración Tributaria",
        licence=AEAT_LICENCE,
        language="es",
    ),
    # ------------------------------------- Commission guides: bot check, download by hand
    Source(
        name="tca_guide_for_traders",
        title="European Commission, EU-UK Trade and Cooperation Agreement: Guide for traders "
        "on rules of origin",
        group="guides",
        suffix=".pdf",
        url="https://taxation-customs.ec.europa.eu/document/download/"
        "2db103b1-08c1-4fdf-9b09-edde7c2dab51_en",
        reference="European Commission (DG TAXUD)",
        licence=EU_LICENCE,
        manual=True,
    ),
    Source(
        name="tca_origin_questions_answers",
        title="European Commission, EU-UK Trade and Cooperation Agreement: compendium of "
        "questions and answers on rules of origin",
        group="guides",
        suffix=".pdf",
        url="https://taxation-customs.ec.europa.eu/system/files/2021-03/"
        "compendium_on_rules_of_origin_q_a.pdf",
        reference="European Commission (DG TAXUD)",
        licence=EU_LICENCE,
        manual=True,
    ),
    Source(
        name="japan_statements_on_origin",
        title="European Commission, EU-Japan EPA: Guidance on statements on origin",
        group="guides",
        suffix=".pdf",
        url="https://taxation-customs.ec.europa.eu/system/files/2019-12/"
        "eu-japan-epa-guidance-statements-on-origin.pdf",
        reference="European Commission (DG TAXUD)",
        licence=EU_LICENCE,
        manual=True,
    ),
    Source(
        name="mercosur_origin_guidance_2026",
        title="European Commission, EU-Mercosur Interim Trade Agreement: Guidance on rules of "
        "origin (April 2026)",
        group="guides",
        suffix=".pdf",
        url="https://taxation-customs.ec.europa.eu/document/download/"
        "a620d284-34db-48d8-99e6-c6cd6252d236_en?filename=EU-Mercosur+ITA+Guidance+RoO+Apr2026.pdf",
        reference="European Commission (DG TAXUD)",
        licence=EU_LICENCE,
        manual=True,
    ),
)


def corpus_index() -> dict[str, tuple[Source, str]]:
    """``{path under rag/corpus/: (source, title)}`` for every file ``download.py`` writes."""
    return {path: (s, title) for s in SOURCES for path, title in s.corpus_files().items()}
