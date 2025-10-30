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
    def extract_low_similarity_examples(student_text: str,
                                    analysis_mode: str,
                                    source_text: str,
                                    refs_list: list[str],
                                    use_semantics: bool,
                                    max_examples: int = 3,
                                    threshold_pct: int = 70):
    """
    Zwraca listę maks. N przykładów o najniższej zgodności:
    [{idx, stud, ref_or_src, score_pct, hint, diff} ...]
    """
    bilingual = analysis_mode.startswith("Dwujęzyczny")
    # Z czego robimy "pool odniesienia"
    if bilingual:
        pool = split_sentences(source_text)
        ref_label = "Źródło"
        ref_text_for_1to1 = source_text
    else:
        # łączymy wszystkie zdania z wielu wzorców
        pool = []
        for r in refs_list:
            pool += split_sentences(r)
        ref_label = "Wzorzec"
        ref_text_for_1to1 = refs_list[0] if refs_list else ""

    if not pool:
        return []

    # Najlepsze dopasowanie zdań (per zdanie studenta)
    rows = sent_align_best(student_text, pool, use_semantics=use_semantics, bilingual=bilingual)

    # Dla rankingu bierzemy semantykę, a jeśli jej brak (nie powinniśmy), to literalność
    def best_score(row):
        if row["sem"] is not None:
            return float(row["sem"]) * 100.0
        if row["lit"] is not None:
            return float(row["lit"]) * 100.0
        return 0.0

    # Posortuj rosnąco i odfiltruj tylko < threshold_pct
    rows_sorted = sorted(rows, key=best_score)
    low_rows = [r for r in rows_sorted if best_score(r) < threshold_pct]

    examples = []
    for r in low_rows[:max_examples]:
        score = int(round(best_score(r)))
        hint = short_hint_for_sentence(
            None if r["lit"] is None else int(round(r["lit"] * 100)),
            None if r["sem"] is None else int(round(r["sem"] * 100)),
            bilingual=bilingual
        )
        examples.append({
            "idx": r["idx"],
            "stud": r["stud"],
            "ref_or_src": r["ref"],
            "score_pct": score,
            "hint": hint,
            "diff": r["diff"]
        })
    return examples, ref_label

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
        # --- Przygotowanie danych ---
        raw = reference_translations.replace("\r\n", "\n")
        refs_list = [blk.strip() for blk in raw.split("\n\n") if blk.strip()] if reference_translations.strip() else []
        if len(refs_list) == 1 and "\n" in refs_list[0]:
            refs_list = [r.strip() for r in raw.split("\n") if r.strip()]

        # --- Similarity (global) ---
        crossling_sim = 0.0
        ref_sim = 0.0
        best_ref_text = ""

        if analysis_mode == "Dwujęzyczny (Źródło ↔ Student)":
            # semantyka źródło↔student
            crossling_sim = crossling_global_similarity(source_text, student_translation) if use_semantics else 0.0
            if include_refs_in_bilingual and refs_list:
                ref_sim, best_ref_text = best_ref_global_similarity(student_translation, refs_list) if use_semantics else (0.0, "")
                # miks: mix_src_refs = waga Źródło↔Student
                auto_similarity = mix_src_refs * crossling_sim + (1.0 - mix_src_refs) * ref_sim
            else:
                auto_similarity = crossling_sim
        else:
            # Tryb wzorcowy (Student ↔ Wzorce)
            ref_sim, best_ref_text = best_ref_global_similarity(student_translation, refs_list) if (use_semantics and refs_list) else (0.0, "")
            lit_sim, lit_best_ref = best_literal_match(student_translation, refs_list) if refs_list else (0.0, "")
            auto_similarity = (0.7 * ref_sim + 0.3 * lit_sim) if use_semantics else lit_sim

        # --- Ocena końcowa ---
        sim_pct = round(auto_similarity * 100, 1)
        faith_pct = round(faithfulness / 5 * 100, 1)
        lang_pct = round(language_quality / 5 * 100, 1)
        style_pct = round(style / 5 * 100, 1)

        # Normalizacja wag
        total_weight = w_auto + w_faith + w_lang + w_style
        w_auto_n = w_auto / total_weight
        w_faith_n = w_faith / total_weight
        w_lang_n = w_lang / total_weight
        w_style_n = w_style / total_weight

        final_pct = round(
            sim_pct * w_auto_n +
            faith_pct * w_faith_n +
            lang_pct * w_lang_n +
            style_pct * w_style_n,
            1
        )

        # Ocena literowa
        if final_pct >= th_5: grade = "5.0"
        elif final_pct >= th_45: grade = "4.5"
        elif final_pct >= th_40: grade = "4.0"
        elif final_pct >= th_35: grade = "3.5"
        elif final_pct >= th_30: grade = "3.0"
        else: grade = "2.0"

        # --- Feedback ---
        feedback_text = generate_feedback(sim_pct, faith_pct, lang_pct, style_pct)

        # --- Zapis do tabeli sesji ---
        new_row = {
            "Data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Student": student_name,
            "Zadanie/Plik": assignment_name,
            "Tryb": analysis_mode,
            "Podobieństwo_crossling": round(crossling_sim * 100, 1),
            "Podobieństwo_wzorcowe": round(ref_sim * 100, 1),
            "Wynik_łączny": sim_pct,
            "Wierność(1-5)": faithfulness,
            "Język(1-5)": language_quality,
            "Styl(1-5)": style,
            "W_auto": w_auto,
            "W_wierność": w_faith,
            "W_język": w_lang,
            "W_styl": w_style,
            "Mix(Źródło↔Wzorce)": round(mix_src_refs, 2) if analysis_mode.startswith("Dwujęzyczny") else None,
            "Progi(%): 5.0": th_5,
            "4.5": th_45,
            "4.0": th_40,
            "3.5": th_35,
            "3.0": th_30,
            "Wynik_finalny_%": final_pct,
            "Ocena": grade
        }

        # Nadpisywanie lub dodawanie
        df = st.session_state.get("results_df", pd.DataFrame(columns=list(new_row.keys())))
        if overwrite_mode:
            mask = (df["Student"] == new_row["Student"]) & (df["Zadanie/Plik"] == new_row["Zadanie/Plik"])
            df = df[~mask]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state.results_df = df

        # --- Wyświetlenie wyników ---
        st.success(f"Wynik końcowy: **{final_pct}% → ocena {grade}**")
        st.markdown("#### 💬 Komentarz automatyczny")
        st.markdown(feedback_text)
        # --- Przykłady niskiej zgodności per zdanie (auto) ---
examples, ref_label = extract_low_similarity_examples(
    student_text=student_translation,
    analysis_mode=analysis_mode,
    source_text=source_text,
    refs_list=refs_list,
    use_semantics=use_semantics,
    max_examples=3,
    threshold_pct=70  # możesz zmienić na 75/80, jeśli chcesz ostrzejsze sito
)

if examples:
    st.markdown("#### 🔎 Przykłady zdań o najniższej zgodności (automatycznie)")
    for ex in examples:
        st.markdown(
            f"**Zdanie {ex['idx']} — {ex['score_pct']}%**\n\n"
            f"- **Student:** {ex['stud']}\n\n"
            f"- **{ref_label}:** {ex['ref_or_src']}\n\n"
            f"- **Wskazówka:** {ex['hint']}\n\n"
        )
        # Diff pokazujemy tylko, gdy to porównanie jednojęzyczne (w dwujęzycznym to mniej użyteczne)
        if not analysis_mode.startswith("Dwujęzyczny") and ex['diff']:
            with st.expander("Podgląd różnic (skrót)"):
                st.code(ex['diff'])
else:
    st.caption("Brak zdań poniżej progu — bardzo równe dopasowanie 👏")


        # Zapamiętaj do panelu zdań
        st.session_state.last_student_translation = student_translation
        st.session_state.last_refs_list = refs_list
        st.session_state.last_use_semantics = use_semantics
        st.session_state.last_analysis_mode = analysis_mode
        st.session_state.last_source_text = source_text
        # ---------- WYNIKI ZBIORCZE + POBIERANIE CSV ----------
st.markdown("---")
st.subheader("📊 Zebrane wyniki (sesja)")

# Upewnij się, że tabela istnieje
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(
        columns=[
            "Data","Student","Zadanie/Plik","Tryb",
            "Podobieństwo_crossling","Podobieństwo_wzorcowe","Wynik_łączny",
            "Wierność(1-5)","Język(1-5)","Styl(1-5)",
            "W_auto","W_wierność","W_język","W_styl",
            "Mix(Źródło↔Wzorce)","Progi(%): 5.0","4.5","4.0","3.5","3.0",
            "Wynik_finalny_%","Ocena"
        ]
    )

df_view = st.session_state.results_df.copy()

# Zamiana wybranych kolumn na % w widoku (bez zmiany w oryginalnym DF)
def _fmt_pct(x):
    if pd.isna(x) or x == "":
        return ""
    try:
        # te kolumny są już w %, ale mogą być floatami; dbamy o całe liczby
        return f"{float(x):.0f}%"
    except:
        return x

for col in ["Podobieństwo_crossling","Podobieństwo_wzorcowe","Wynik_łączny","Wynik_finalny_%"]:
    if col in df_view.columns:
        df_view[col] = df_view[col].apply(_fmt_pct)

st.dataframe(df_view, use_container_width=True)

# Średnie (tylko jeśli są dane liczbowe)
if not st.session_state.results_df.empty:
    # Bezpieczne rzutowanie
    def _to_float(series):
        return pd.to_numeric(series, errors="coerce")

    mean_final = _to_float(st.session_state.results_df["Wynik_finalny_%"]).mean()
    # Ocena jako float (np. "4.5" → 4.5)
    mean_grade = _to_float(st.session_state.results_df["Ocena"]).mean()

    st.markdown("### 📈 Średnie (sesja)")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Średni wynik ( % )", f"{mean_final:.0f}%" if not pd.isna(mean_final) else "—")
    with c2:
        st.metric("Średnia ocena (PL)", f"{mean_grade:.1f}" if not pd.isna(mean_grade) else "—")

# Pobieranie CSV (oryginalny DF, bez formatowania %)
csv_data = st.session_state.results_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Pobierz wyniki jako CSV",
    data=csv_data,
    file_name="wyniki_tlumaczen.csv",
    mime="text/csv"
)

