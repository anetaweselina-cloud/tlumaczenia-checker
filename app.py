import streamlit as st
import difflib
import pandas as pd
import numpy as np
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer

# (Opcjonalnie) Google Sheets
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GS_AVAILABLE = True
except Exception:
    GS_AVAILABLE = False

# ---------- USTAWIENIA STRONY ----------
st.set_page_config(page_title="Ocena tłumaczeń – wersja nauczycielska (dwujęzyczna)", layout="wide")

# ---------- INICJALIZACJA STANU (PAMIĘĆ UI) ----------
def _init_state():
    ss = st.session_state
    # Panel zdań – pamięć
    ss.setdefault("sent_mode", "Najlepsze dopasowanie")  # albo "1:1 alignment"
    ss.setdefault("low_thr", 70)                         # próg filtrowania zdań (%)
    ss.setdefault("show_only_low", True)
    # Ostatnie dane do panelu zdań
    ss.setdefault("last_student_translation", "")
    ss.setdefault("last_refs_list", [])
    ss.setdefault("last_use_semantics", True)
    ss.setdefault("last_analysis_mode", "Dwujęzyczny (Źródło ↔ Student)")
    ss.setdefault("last_source_text", "")
    # Tabela wyników
    if "results_df" not in ss:
        ss.results_df = pd.DataFrame(
            columns=[
                "Data","Student","Zadanie/Plik",
                "Tryb","Podobieństwo_crossling","Podobieństwo_wzorcowe","Wynik_łączny",
                "Wierność(1-5)","Język(1-5)","Styl(1-5)",
                "W_auto","W_wierność","W_język","W_styl",
                "Mix(Źródło↔Wzorce)","Progi(%): 5.0","4.5","4.0","3.5","3.0",
                "Wynik_finalny_%","Ocena"
            ]
        )

_init_state()

# ---------- MODEL SEMANTYCZNY (wielojęzyczny) ----------
@st.cache_resource
def load_st_model():
    # Model wielojęzyczny do PL↔EN (sprawdza się także dla EN↔EN / PL↔PL)
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ---------- POMOCNICZE: podobieństwa globalne ----------
def cosine_sim(a, b):
    return float(np.dot(a, b))

def embed_texts(texts: list[str]):
    model = load_st_model()
    return model.encode(texts, normalize_embeddings=True)

def crossling_global_similarity(source_text: str, student_text: str) -> float:
    """Semantyczne podobieństwo międzyjęzykowe źródło↔tłumaczenie studenta (0..1)."""
    if not source_text.strip() or not student_text.strip():
        return 0.0
    a, b = embed_texts([source_text, student_text])
    return cosine_sim(a, b)

def best_ref_global_similarity(student_text: str, refs: list[str]) -> tuple[float,str]:
    """Najlepsze semantyczne dopasowanie student↔wzorce (0..1, ref_text)."""
    if not refs:
        return 0.0, ""
    model = load_st_model()
    emb_s = model.encode(student_text, normalize_embeddings=True)
    emb_r = model.encode(refs, normalize_embeddings=True)
    sims = [float(np.dot(emb_s, r)) for r in emb_r]
    k = int(np.argmax(sims))
    return sims[k], refs[k]

def best_literal_match(student_text: str, refs: list[str]):
    """Literalne dopasowanie (używamy tylko informacyjnie)."""
    best_ref, best_score = "", 0.0
    for ref in refs:
        score = difflib.SequenceMatcher(None, student_text.lower(), ref.lower()).ratio()
        if score > best_score:
            best_score, best_ref = score, ref
    return best_score, best_ref

# ---------- KOMENTARZE SZABLONOWE ----------
SECTION_TEMPLATES = {
    "auto": {
        "name": "Analiza automatyczna (Similarity)",
        "ranges": [
            (90, "Bardzo wysoka zgodność znaczeniowa. Parafrazy trafne i adekwatne do kontekstu."),
            (80, "Wysoka zgodność; drobne różnice w ujęciu treści, ale sens zachowany."),
            (70, "Umiarkowana zgodność; miejscami przesunięcie sensu."),
            (60, "Niska zgodność; sprawdź kompletność informacji względem źródła/wzorców."),
            (0,  "Bardzo niska zgodność; przeanalizuj segment po segmencie.")
        ],
    },
    "faith": {
        "name": "Wierność",
        "ranges": [
            (90, "Wierność bardzo dobra — brak istotnych pominięć i nadinterpretacji."),
            (80, "Generalnie wierne; drobne skróty lub uogólnienia."),
            (70, "Częściowo wierne; doprecyzuj szczegóły i terminy kluczowe."),
            (60, "Niska wierność; część treści zmieniona lub pominięta."),
            (0,  "Wierność bardzo niska; porównaj zdania 1:1 ze źródłem.")
        ],
    },
    "lang": {
        "name": "Poprawność językowa",
        "ranges": [
            (90, "Bardzo dobra poprawność: gramatyka/składnia bez zarzutu, naturalne kolokacje."),
            (80, "Dobra poprawność; sporadyczne potknięcia nie zaburzają odbioru."),
            (70, "Umiarkowana poprawność; popraw drobne błędy gramatyczne i kolokacyjne."),
            (60, "Niska poprawność; widoczne kalki i błędy składniowe."),
            (0,  "Bardzo niska poprawność; zalecana gruntowna redakcja.")
        ],
    },
    "style": {
        "name": "Styl / naturalność",
        "ranges": [
            (90, "Styl bardzo naturalny i spójny z rejestrem."),
            (80, "Styl dobry; miejscami sztywność lub dosłowność."),
            (70, "Styl umiarkowany; uprość składnię i stosuj idiomatyczne frazy."),
            (60, "Styl słaby; widoczne kaleki i sztuczne brzmienie."),
            (0,  "Styl bardzo słaby; przeformułuj dla płynności i rejestru.")
        ],
    },
}

def _comment_from_templates(section_key: str, pct: float) -> str:
    tmpl = SECTION_TEMPLATES[section_key]
    for threshold, text in tmpl["ranges"]:
        if pct >= threshold:
            return f"**{tmpl['name']}** — {text}"
    return f"**{tmpl['name']}** — (brak szablonu)"

def generate_feedback(sim_pct: float, faith_pct: float, lang_pct: float, style_pct: float) -> str:
    parts = [
        _comment_from_templates("auto",  sim_pct),
        _comment_from_templates("faith", faith_pct),
        _comment_from_templates("lang",  lang_pct),
        _comment_from_templates("style", style_pct),
    ]
    weakest_label, _ = min(
        [("wierność", faith_pct), ("język", lang_pct), ("styl", style_pct), ("similarity", sim_pct)],
        key=lambda x: x[1]
    )
    tips = {
        "wierność": "Porównaj tłumaczenie ze źródłem zdanie po zdaniu; sprawdź kompletność i terminologię.",
        "język": "Redakcja gramatyczno-kolokacyjna; przeczytaj tekst na głos.",
        "styl": "Uprość składnię i stosuj idiomatyczne frazy; dopasuj rejestr.",
        "similarity": "Skup się na zdaniach o najniższym dopasowaniu (panel poniżej).",
    }
    summary = f"\n**Podsumowanie:** najsłabszy aspekt: **{weakest_label}**. Sugestia: {tips[weakest_label]}"
    return "\n\n".join(parts) + summary

# ---------- PORÓWNANIE ZDAŃ ----------
_SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')

def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if not re.search(r'[.!?]', text):
        return [text]
    parts = _SENT_SPLIT_RE.split(text)
    merged = []
    for p in parts:
        if merged and len(p.strip()) < 2:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p.strip())
    return [s for s in merged if s]

def sent_align_best(student_text: str, pool_ref_sents: list[str], use_semantics: bool, bilingual: bool):
    """Najlepsze dopasowanie zdań: gdy bilingual=True – pool_ref_sents to zdania ŹRÓDŁOWE."""
    if not pool_ref_sents:
        return []
    stud_sents = split_sentences(student_text)
    model = load_st_model() if (use_semantics or bilingual) else None
    rows = []
    # Pre-embed pool dla szybkości
    emb_pool = model.encode(pool_ref_sents, normalize_embeddings=True) if (model and (use_semantics or bilingual)) else None

    for i, ss in enumerate(stud_sents, start=1):
        # Literalność sensowna tylko gdy ten sam język (czyli tryb wzorcowy)
        best_lit, best_lit_ref = 0.0, ""
        if not bilingual:
            for rs in pool_ref_sents:
                sc = difflib.SequenceMatcher(None, ss.lower(), rs.lower()).ratio()
                if sc > best_lit:
                    best_lit, best_lit_ref = sc, rs

        # Semantyka – działa w obu trybach (wielojęzycznie)
        if model and emb_pool is not None and ss.strip():
            emb_ss = model.encode(ss, normalize_embeddings=True)
            sims = [float(np.dot(emb_ss, r)) for r in emb_pool]
            k = int(np.argmax(sims))
            best_sem, best_sem_ref = sims[k], pool_ref_sents[k]
        else:
            best_sem, best_sem_ref = None, None

        # Diff podgląd – sensowniejszy literalnie (gdy nie bilingual)
        if not bilingual:
            base_ref_for_diff = best_lit_ref or ""
            diff_tokens = list(difflib.ndiff(ss.split(), base_ref_for_diff.split()))
            if len(diff_tokens) > 120:
                diff_tokens = diff_tokens[:120] + ["..."]
            diff_preview = " ".join(diff_tokens)
        else:
            diff_preview = "(porównanie międzyjęzykowe – diff literalny pominięty)"

        shown_ref = best_sem_ref if (best_sem_ref) else best_lit_ref
        rows.append({
            "idx": i,
            "stud": ss,
            "ref": shown_ref,
            "lit": None if bilingual else best_lit,
            "sem": best_sem,
            "diff": diff_preview,
        })
    return rows

def sent_align_1to1(student_text: str, ref_text: str, use_semantics: bool, bilingual: bool):
    """1:1: i-te zdanie studenta ↔ i-te zdanie z ref_text (źródło lub główny wzorzec)."""
    stud_sents = split_sentences(student_text)
    ref_sents  = split_sentences(ref_text)
    n = max(len(stud_sents), len(ref_sents))
    model = load_st_model() if (use_semantics or bilingual) else None
    rows = []
    for i in range(1, n+1):
        ss = stud_sents[i-1] if i-1 < len(stud_sents) else ""
        rs = ref_sents[i-1]  if i-1 < len(ref_sents)  else ""
        # literalność tylko w trybie wzorcowym
        best_lit = None if bilingual else (difflib.SequenceMatcher(None, ss.lower(), rs.lower()).ratio() if ss and rs else 0.0)
        if model and ss and rs:
            emb_ss = model.encode(ss, normalize_embeddings=True)
            emb_rs = model.encode(rs, normalize_embeddings=True)
            best_sem = float(np.dot(emb_ss, emb_rs))
        else:
            best_sem = None
        if not bilingual:
            diff_tokens = list(difflib.ndiff(ss.split(), rs.split()))
            if len(diff_tokens) > 120:
                diff_tokens = diff_tokens[:120] + ["..."]
            diff_preview = " ".join(diff_tokens)
        else:
            diff_preview = "(porównanie międzyjęzykowe – diff literalny pominięty)"
        rows.append({"idx": i, "stud": ss, "ref": rs, "lit": best_lit, "sem": best_sem, "diff": diff_preview})
    return rows

def short_hint_for_sentence(lit_pct: float|None, sem_pct: float|None, bilingual: bool) -> str:
    s = sem_pct if sem_pct is not None else (lit_pct if lit_pct is not None else 0)
    if s >= 80: return "OK – zgodność wysoka."
    if s >= 70: return "Drobne rozbieżności – doprecyzuj szczegóły."
    if s >= 60: return "Sprawdź sens względem odniesienia; rozważ przeformułowanie."
    return "Niska zgodność – porównaj ze źródłem/wzorcem, uprość składnię i doprecyzuj słownictwo."

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### ⚙️ Instrukcja")
    st.markdown(
        "- Wklej **tekst źródłowy** (PL/EN), **tłumaczenie studenta**.\n"
        "- (Opcjonalnie) w trybie dwujęzycznym możesz **dołączyć wzorce** i ustawić miks.\n"
        "- Uzupełnij **rubrykę** i kliknij **Oceń tłumaczenie**.\n"
        "- Panel zdań działa na **ostatnio ocenionych** tekstach."
    )
    st.markdown("---")
    st.markdown("#### Zapisywanie wyników")
    overwrite_mode = st.toggle("Nadpisuj istniejący wpis (Student+Zadanie)", value=True)
    st.markdown("#### Google Sheets (opcjonalnie)")
    st.caption("Działa, jeśli skonfigurujesz sekrety w Streamlit Cloud (service account + sheet_id).")
    use_gsheets = st.toggle("Włącz zapis do Google Sheets", value=False)

st.title("📘 Ocena tłumaczeń — tryb dwujęzyczny i wzorcowy")

# ---------- TRYB ANALIZY ----------
analysis_mode = st.radio(
    "Tryb analizy",
    options=["Dwujęzyczny (Źródło ↔ Student)", "Wzorcowy (Student ↔ Wzorce)"],
    index=0,
    help="Dwujęzyczny: porównanie źródło↔student (PL↔EN). Wzorcowy: student↔tłumaczenia wzorcowe."
)

# ---------- FORMULARZ ----------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Dane wejściowe")
    student_name = st.text_input("👤 Imię i nazwisko studenta", "")
    assignment_name = st.text_input("🗂️ Nazwa zadania / pliku (opcjonalnie)", "")
    source_text = st.text_area("🧾 Tekst źródłowy (PL/EN)", height=120)
    student_translation = st.text_area("✍️ Tłumaczenie studenta (PL/EN)", height=180)

    # Wzorce – zawsze dostępne, ale mogą być pomijane
    reference_translations = st.text_area(
        "✅ Tłumaczenia wzorcowe (opcjonalnie, każde tłumaczenie oddziel **pustą linią**)",
        height=180,
        placeholder=("Wklej 0, 1 lub więcej poprawnych tłumaczeń.\n"
                     "Każde tłumaczenie oddziel pustą linią (Enter, Enter).")
    )

with col_right:
    st.subheader("Rubryka oceny")
    faithfulness = st.slider("Wierność (1–5)", 1, 5, 4)
    language_quality = st.slider("Poprawność językowa (1–5)", 1, 5, 4)
    style = st.slider("Styl / naturalność (1–5)", 1, 5, 4)

    st.subheader("Analiza automatyczna")
    use_semantics = st.toggle("Użyj analizy semantycznej (rekomendowane)", value=True)

    # Miks tylko w dwujęzycznym, jeśli użytkownik chce uwzględnić wzorce
    include_refs_in_bilingual = False
    mix_src_refs = 1.0
    if analysis_mode == "Dwujęzyczny (Źródło ↔ Student)":
        include_refs_in_bilingual = st.checkbox("Uwzględnij wzorce dodatkowo (student↔wzorce)", value=False,
                                                help="Jeśli zaznaczysz i podasz wzorce, Similarity będzie mieszanką: Źródło↔Student oraz Student↔Wzorce.")
        if include_refs_in_bilingual:
            mix_src_refs = st.slider("Mix Similarity: Źródło (lewo) ↔ Wzorce (prawo)", 0.0, 1.0, 0.7, 0.05,
                                     help="0.7 = 70% wagi dla Źródło↔Student, 30% dla Student↔Wzorce")

    st.subheader("Wagi komponentów")
    w_auto = st.number_input("Waga: Similarity", min_value=0.0, value=0.40, step=0.05)
    w_faith = st.number_input("Waga: Wierność",   min_value=0.0, value=0.35, step=0.05)
    w_lang  = st.number_input("Waga: Język",      min_value=0.0, value=0.15, step=0.05)
    w_style = st.number_input("Waga: Styl",       min_value=0.0, value=0.10, step=0.05)
    st.caption("Wagi nie muszą sumować się do 1 — znormalizujemy je automatycznie.")

# ---------- SKALA OCEN ----------
st.markdown("### 🧮 Skala ocen (progi % → ocena PL)")
cols = st.columns(5)
with cols[0]: th_5  = st.number_input("5.0 od (%)", 0.0, 100.0, 90.0, 1.0)
with cols[1]: th_45 = st.number_input("4.5 od (%)", 0.0, 100.0, 80.0, 1.0)
with cols[2]: th_40 = st.number_input("4.0 od (%)", 0.0, 100.0, 70.0, 1.0)
with cols[3]: th_35 = st.number_input("3.5 od (%)", 0.0, 100.0, 60.0, 1.0)
with cols[4]: th_30 = st.number_input("3.0 od (%)", 0.0, 100.0, 50.0, 1.0)
st.caption("Upewnij się, że progi maleją: 5.0 ≥ 4.5 ≥ 4.0 ≥ 3.5 ≥ 3.0")

# ---------- PRZYCISK: ANALIZA + ZAPAMIĘTANIE ----------
if st.button("🔎 Oceń tłumaczenie", type="primary"):
    if not student_translation.strip():
        st.error("Wprowadź tłumaczenie studenta.")
    elif analysis_mode == "Dwujęzyczny (Źródło ↔ Student)" and not source_text.strip():
        st.error("W trybie dwujęzycznym wymagany jest tekst źródłowy.")
    else:
        # Wzorce – przygotuj listę (0..N)
        raw = reference_translations.replace("\r\n", "\n")
        refs_list = [blk.strip() for blk in raw.split("\n\n") if blk.strip()] if reference_translations.strip() else []
        if len(refs_list) == 1 and "\n" in refs_list[0]:
            refs_list = [r.strip() for r in raw.split("\n") if r.strip()]

        # --- Similarity (global) ---
        crossling_sim = None
        ref_sim = None
        best_ref_lit_sim = None
        best_ref_text = ""

        if analysis_mode == "Dwujęzyczny (Źródło ↔ Student)":
            crossling_sim = crossling_global_similarity(source_text, student_translation) if use_semantics else 0.0
            if include_refs_in_bilingual and refs_list:
                ref_sim, best_ref_text = best_ref_global_similarity(student_translation, refs_list) if use_semantics else (0.0, "")
                # Miks: mix_src_refs to waga Źródła
                auto_similarity = mix_src_refs * crossling_sim + (1.0 - mix_src_refs) * ref_sim
            else:
                auto_similarity = crossling_sim
        else:
    # Tryb wzorcowy (Student ↔ Wzorce)
    ref_sim, best_ref_text = best_ref_global_similarity(student_translation, refs_list) if (use_semantics and refs_list) else (0.0, "")
    lit_sim, lit_best_ref = best_literal_match(student_translation, refs_list) if refs_list else (0.0, "")
    auto_similarity = (0.7 * ref_sim + 0.3 * lit_sim) if use_semantics else lit_sim

               
