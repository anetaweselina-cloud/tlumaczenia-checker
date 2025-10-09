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
st.set_page_config(page_title="Ocena tłumaczeń – wersja nauczycielska", layout="wide")

# ---------- MODEL SEMANTYCZNY ----------
@st.cache_resource
def load_st_model():
    # Lekki model do porównań zdań (ok. 22 MB)
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def best_semantic_match(student_text: str, refs: list[str]):
    """Zwraca (best_score (0..1), best_ref)."""
    model = load_st_model()
    emb_student = model.encode(student_text, normalize_embeddings=True)
    emb_refs = model.encode(refs, normalize_embeddings=True)
    sims = [float(np.dot(emb_student, r)) for r in emb_refs]  # dot == cosine
    idx = int(np.argmax(sims))
    return sims[idx], refs[idx]

# ---------- POMOCNICZE (tekst globalnie) ----------
def compute_best_match(student_text: str, refs: list[str]):
    """Literalne podobieństwo; zwraca (best_score, best_ref, diff_preview_text)."""
    best_ref, best_score = "", 0.0
    for ref in refs:
        score = difflib.SequenceMatcher(None, student_text.lower(), ref.lower()).ratio()
        if score > best_score:
            best_score, best_ref = score, ref
    diff_tokens = list(difflib.ndiff(student_text.split(), (best_ref or "").split()))
    if len(diff_tokens) > 200:
        diff_tokens = diff_tokens[:200] + ["..."]
    diff_preview = " ".join(diff_tokens)
    return best_score, best_ref, diff_preview

def ensure_results_df():
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame(
            columns=[
                "Data","Student","Zadanie/Plik",
                "Podobieństwo_literalne","Podobieństwo_semantyczne","Wynik_łączny",
                "Wierność(1-5)","Język(1-5)","Styl(1-5)",
                "W_auto","W_wierność","W_język","W_styl",
                "Progi(%): 5.0","4.5","4.0","3.5","3.0",
                "Wynik_finalny_%","Ocena"
            ]
        )

def grade_from_thresholds(pct: float, th_5, th_45, th_40, th_35, th_30) -> str:
    if pct >= th_5:  return "5.0"
    if pct >= th_45: return "4.5"
    if pct >= th_40: return "4.0"
    if pct >= th_35: return "3.5"
    if pct >= th_30: return "3.0"
    return "2.0"

def gs_get_client_from_secrets():
    sa_info = st.secrets.get("gcp_service_account", None)
    sheet_id = st.secrets.get("sheet_id", None)
    if not sa_info or not sheet_id:
        return None, None
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    client = gspread.authorize(creds)
    return client, sheet_id

def gs_append_row(row_list: list):
    client, sheet_id = gs_get_client_from_secrets()
    if not client or not sheet_id:
        raise RuntimeError("Brak konfiguracji Google Sheets w st.secrets.")
    sh = client.open_by_key(sheet_id)
    ws = sh.sheet1
    ws.append_row(row_list, value_input_option="USER_ENTERED")

# ---------- KOMENTARZE SZABLONOWE (per sekcja) ----------
SECTION_TEMPLATES = {
    "auto": {
        "name": "Analiza automatyczna (Similarity)",
        "ranges": [
            (90, "Bardzo wysoka zgodność znaczeniowa z wzorcami. Parafrazy trafne i adekwatne do kontekstu."),
            (80, "Wysoka zgodność; drobne różnice w ujęciu treści, ale sens zachowany."),
            (70, "Umiarkowana zgodność; część fragmentów parafrazowana z przesunięciem sensu."),
            (60, "Niska zgodność; sprawdź, czy wszystkie informacje źródła zostały zachowane."),
            (0,  "Bardzo niska zgodność; przeanalizuj segment po segmencie względem źródła i wzorców.")
        ],
    },
    "faith": {
        "name": "Wierność",
        "ranges": [
            (90, "Wierność bardzo dobra — brak istotnych pominięć i nadinterpretacji."),
            (80, "Generalnie wierne; drobne skróty lub uogólnienia."),
            (70, "Częściowo wierne; doprecyzuj szczegóły i terminy kluczowe."),
            (60, "Niska wierność; część treści zmieniona lub pominięta."),
            (0,  "Wierność bardzo niska; warto porównać zdania 1:1 ze źródłem.")
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
            (70, "Styl umiarkowany; rozważ prostszą składnię i idiomatyczne frazy."),
            (60, "Styl słaby; widoczne kaleki i sztuczne brzmienie."),
            (0,  "Styl bardzo słaby; zalecane przeformułowanie dla płynności i rejestru.")
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
    weakest_label, weakest_score = min(
        [("wierność", faith_pct), ("język", lang_pct), ("styl", style_pct), ("similarity", sim_pct)],
        key=lambda x: x[1]
    )
    tips = {
        "wierność": "Porównaj tłumaczenie ze źródłem zdanie po zdaniu; sprawdź kompletność informacji i terminologię.",
        "język": "Zrób redakcję gramatyczno-kolokacyjną; przeczytaj tekst na głos.",
        "styl": "Uprość składnię i wybierz bardziej idiomatyczne frazy; dopasuj rejestr.",
        "similarity": "Zestaw tłumaczenie z kilkoma wzorcami; doprecyzuj fragmenty o najniższym dopasowaniu.",
    }
    summary = f"\n**Podsumowanie:** najmocniej warto popracować nad: **{weakest_label}**. Sugestia: {tips[weakest_label]}"
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

def _best_literal(ss: str, pool_ref_sents: list[str]):
    best_lit, best_lit_ref = 0.0, ""
    for rs in pool_ref_sents:
        sc = difflib.SequenceMatcher(None, ss.lower(), rs.lower()).ratio()
        if sc > best_lit:
            best_lit, best_lit_ref = sc, rs
    return best_lit, best_lit_ref

def _best_semantic(ss: str, pool_ref_sents: list[str]):
    model = load_st_model()
    emb_ss = model.encode(ss, normalize_embeddings=True)
    emb_pool = model.encode(pool_ref_sents, normalize_embeddings=True)
    sims = [float(np.dot(emb_ss, r)) for r in emb_pool]
    k = int(np.argmax(sims))
    return sims[k], pool_ref_sents[k]

def sent_level_alignment_best(student_text: str, refs_list: list[str], use_semantics: bool):
    """Dla każdego zdania studenta: najlepszy wzorzec z całej puli zdań referencyjnych."""
    pool_ref_sents = []
    for ref in refs_list:
        for rs in split_sentences(ref):
            if rs.strip():
                pool_ref_sents.append(rs.strip())
    if not pool_ref_sents:
        return []

    stud_sents = split_sentences(student_text)
    rows = []
    for i, ss in enumerate(stud_sents, start=1):
        best_lit, best_lit_ref = _best_literal(ss, pool_ref_sents)
        if use_semantics:
            best_sem, best_sem_ref = _best_semantic(ss, pool_ref_sents)
        else:
            best_sem, best_sem_ref = None, None

        diff_tokens = list(difflib.ndiff(ss.split(), (best_lit_ref or "").split()))
        if len(diff_tokens) > 120:
            diff_tokens = diff_tokens[:120] + ["..."]
        diff_preview = " ".join(diff_tokens)
        shown_ref = best_sem_ref if (use_semantics and best_sem_ref) else best_lit_ref

        rows.append({
            "idx": i,
            "stud": ss,
            "ref": shown_ref,
            "lit": best_lit,        # 0..1
            "sem": best_sem,        # 0..1 lub None
            "diff": diff_preview,
        })
    return rows

def sent_level_alignment_1to1(student_text: str, refs_list: list[str], use_semantics: bool):
    """Parowanie 1:1 wg kolejności zdań: zdanie i-te studenta ↔ zdanie i-te z pierwszego wzorca."""
    primary_ref = refs_list[0] if refs_list else ""
    stud_sents = split_sentences(student_text)
    ref_sents  = split_sentences(primary_ref)
    n = max(len(stud_sents), len(ref_sents))
    rows = []
    model = load_st_model() if use_semantics else None

    for i in range(1, n+1):
        ss = stud_sents[i-1] if i-1 < len(stud_sents) else ""
        rs = ref_sents[i-1]  if i-1 < len(ref_sents)  else ""

        best_lit = difflib.SequenceMatcher(None, ss.lower(), rs.lower()).ratio() if ss and rs else 0.0
        if use_semantics and ss and rs:
            emb_ss = model.encode(ss, normalize_embeddings=True)
            emb_rs = model.encode(rs, normalize_embeddings=True)
            best_sem = float(np.dot(emb_ss, emb_rs))
        else:
            best_sem = None

        diff_tokens = list(difflib.ndiff(ss.split(), rs.split()))
        if len(diff_tokens) > 120:
            diff_tokens = diff_tokens[:120] + ["..."]
        diff_preview = " ".join(diff_tokens)

        rows.append({
            "idx": i,
            "stud": ss,
            "ref": rs,
            "lit": best_lit,
            "sem": best_sem,
            "diff": diff_preview,
        })
    return rows

def short_hint_for_sentence(lit_pct: float, sem_pct: float) -> str:
    """Krótka wskazówka dla zdań o niskich wynikach."""
    s = sem_pct if sem_pct is not None else lit_pct
    if s >= 80:
        return "OK – zgodność wysoka."
    if s >= 70:
        return "Drobne rozbieżności – doprecyzuj szczegóły."
    if s >= 60:
        return "Rozważ przeformułowanie fragmentu lub sprawdź sens względem źródła."
    return "Niska zgodność – porównaj ze źródłem, uprość składnię i doprecyzuj słownictwo."

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### ⚙️ Instrukcja")
    st.markdown(
        "- Wklej **tekst źródłowy** (opcjonalnie), **tłumaczenie studenta** i **kilka wzorców**.\n"
        "- **Każde tłumaczenie wzorcowe oddziel pustą linią** (Enter, Enter).\n"
        "- Uzupełnij **rubrykę** i kliknij **Oceń tłumaczenie**.\n"
        "- Na dole pobierzesz **CSV**. (Opcjonalnie: zapis do **Google Sheets** w sekcjach App Settings)."
    )
    st.caption("Similarity = miks semantyki i dopasowania literalnego. Wagi i progi ustawisz niżej.")
    st.markdown("---")
    st.markdown("#### Google Sheets (opcjonalnie)")
    st.caption("Działa, jeśli skonfigurujesz sekrety w Streamlit Cloud (service account + sheet_id).")
    use_gsheets = st.toggle("Włącz zapis do Google Sheets", value=False)

st.title("📘 Ocena tłumaczeń studentów — wersja nauczycielska")

# ---------- FORMULARZ ----------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Dane wejściowe")
    student_name = st.text_input("👤 Imię i nazwisko studenta", "")
    assignment_name = st.text_input("🗂️ Nazwa zadania / pliku (opcjonalnie)", "")
    source_text = st.text_area("🧾 Tekst źródłowy (dla kontekstu, opcjonalnie)", height=120)
    student_translation = st.text_area("✍️ Tłumaczenie studenta", height=180)
    reference_translations = st.text_area(
        "✅ Tłumaczenia wzorcowe (każde tłumaczenie oddziel **pustą linią**)",
        height=200,
        placeholder=("Wklej jedno lub więcej poprawnych tłumaczeń.\n"
                     "Każde tłumaczenie oddziel pustą linią (Enter, Enter).")
    )

with col_right:
    st.subheader("Rubryka oceny")
    faithfulness = st.slider("Wierność (1–5)", 1, 5, 4)
    language_quality = st.slider("Poprawność językowa (1–5)", 1, 5, 4)
    style = st.slider("Styl / naturalność (1–5)", 1, 5, 4)

    st.subheader("Analiza (globalna)")
    use_semantics = st.toggle("Użyj analizy semantycznej (rekomendowane)", value=True)
    sem_weight_mix = st.slider("Waga semantyki w Similarity", 0.0, 1.0, 0.7, 0.05)

    st.subheader("Wagi komponentów (do oceny końcowej)")
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

# ---------- PRZYCISK ----------
if st.button("🔎 Oceń tłumaczenie", type="primary"):
    if not student_translation.strip():
        st.error("Wprowadź tłumaczenie studenta.")
    elif not reference_translations.strip():
        st.error("Wprowadź co najmniej jedno tłumaczenie wzorcowe.")
    else:
        # Lista wzorców (puste linie rozdzielają tłumaczenia)
        raw = reference_translations.replace("\r\n", "\n")
        refs_list = [blk.strip() for blk in raw.split("\n\n") if blk.strip()]
        if len(refs_list) == 1 and "\n" in refs_list[0]:
            refs_list = [r.strip() for r in raw.split("\n") if r.strip()]

        # Podobieństwo globalne
        lit_sim, lit_best_ref, diff_preview = compute_best_match(student_translation, refs_list)
        if use_semantics:
            sem_sim, sem_best_ref = best_semantic_match(student_translation, refs_list)
        else:
            sem_sim, sem_best_ref = None, None
        display_best_ref = sem_best_ref if (use_semantics and sem_best_ref) else lit_best_ref
        combined_similarity = (sem_weight_mix * sem_sim + (1 - sem_weight_mix) * lit_sim) if use_semantics else lit_sim

        # Prezentacja metryk globalnych
        st.success("Analiza zakończona.")
        st.markdown("#### Najbliższe tłumaczenie wzorcowe")
        st.write(display_best_ref if display_best_ref else "—")
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Podobieństwo (literalne)", f"{lit_sim:.0%}")
        with m2: st.metric("Podobieństwo (semantyczne)", "—" if sem_sim is None else f"{sem_sim:.0%}")
        with m3: st.metric("Similarity (łączna)", f"{combined_similarity:.0%}")
        with st.expander("Podgląd różnic (skrót – wobec wzorca literalnego)"):
            st.code(diff_preview)

        # ---------- PANEL PORÓWNANIA ZDAŃ ----------
        st.markdown("### 🔎 Porównanie zdań (zdanie-za-zdaniem)")
        mode_col, thr_col, chk_col = st.columns([1.2, 1, 1])
        with mode_col:
            sent_mode = st.radio(
                "Tryb dopasowania zdań",
                options=["Najlepsze dopasowanie", "1:1 alignment"],
                help="‘Najlepsze’ wybiera dla każdego zdania studenta najlepiej pasujące zdanie z całej puli wzorców.\n‘1:1’ łączy zdanie i-te studenta ze zdaniem i-tym pierwszego wzorca."
            )
        with thr_col:
            low_thr = st.slider("Próg filtrowania (%)", 0, 100, 70, 5, help="Pokaż tylko zdania poniżej tego progu.")
        with chk_col:
            show_only_low = st.checkbox("Pokaż tylko poniżej progu", value=True)

        # Oblicz wiersze porównań
        if sent_mode == "1:1 alignment":
            rows = sent_level_alignment_1to1(student_translation, refs_list, use_semantics)
        else:
            rows = sent_level_alignment_best(student_translation, refs_list, use_semantics)

        # Zbuduj tabelę + kolorowanie i komentarze per zdanie
        table_rows = []
        for r in rows:
            lit_pct = int(round(r["lit"] * 100)) if r["lit"] is not None else None
            sem_pct = int(round(r["sem"] * 100)) if r["sem"] is not None else None
            shown = sem_pct if sem_pct is not None else lit_pct
            if show_only_low and shown is not None and shown >= low_thr:
                continue
            hint = short_hint_for_sentence(lit_pct if lit_pct is not None else 0,
                                           sem_pct if sem_pct is not None else None)
            table_rows.append({
                "Zdanie #": r["idx"],
                "Zdanie studenta": r["stud"],
                "Najlepszy wzorzec (zdanie)" if sent_mode!="1:1 alignment" else "Wzorzec (1:1)": r["ref"],
                "Literalnie (%)": lit_pct,
                "Semantycznie (%)": sem_pct if sem_pct is not None else "",
                "Różnice (skrót)": r["diff"],
                "Wskazówka": hint,
            })

        if not table_rows:
            st.info("Brak zdań do wyświetlenia przy wybranych filtrach/trybie.")
        else:
            df_sent = pd.DataFrame(table_rows)

            # Funkcja kolorująca komórki wg progów
            def color_thresholds(val):
                if val == "" or pd.isna(val):
                    return ""
                try:
                    v = float(val)
                except:
                    return ""
                if v >= 80:
                    return "background-color: #e6ffe6"  # zielonkawe
                if v >= 60:
                    return "background-color: #fff9cc"  # żółtawe
                return "background-color: #ffe6e6"      # czerwonawe

            # Styler – kolorujemy dwie kolumny procentowe
            styled = df_sent.style.applymap(color_thresholds, subset=["Literalnie (%)", "Semantycznie (%)"])
            st.dataframe(styled, use_container_width=True)

        # ---------- HINTY OGÓLNE (miękkie) ----------
        issues = []
        text_lower = student_translation.lower()
        refs_joined_lower = " ".join(refs_list).lower()
        if "since" in text_lower and "for" in refs_joined_lower:
            issues.append("Możliwe nadużycie 'since' dla okresu czasu – rozważ 'for'.")
        if "make a photo" in text_lower:
            issues.append("Kolokacja: zwykle 'take a photo', nie 'make a photo'.")
        if "i have" in text_lower and "years old" in text_lower:
            issues.append("Kalka: 'I am X years old', nie 'I have X years old'.")
        if issues:
            st.markdown("#### Potencjalne kwestie do sprawdzenia (ogólne)")
            for it in issues:
                st.write(f"- {it}")

        # ---------- OCENA KOŃCOWA ----------
        # Wagi → normalizacja
        w_sum = max(w_auto + w_faith + w_lang + w_style, 1e-9)
        wn_auto, wn_faith, wn_lang, wn_style = w_auto/w_sum, w_faith/w_sum, w_lang/w_sum, w_style/w_sum
        # Skale (0..1)
        faith_norm = faithfulness / 5.0
        lang_norm  = language_quality / 5.0
        style_norm = style / 5.0
        final_0_1 = wn_auto * combined_similarity + wn_faith * faith_norm + wn_lang * lang_norm + wn_style * style_norm
        final_pct = float(final_0_1 * 100.0)
        final_grade = grade_from_thresholds(final_pct, th_5, th_45, th_40, th_35, th_30)

        # Komentarz łączny zależny od % w sekcjach
        sim_pct = combined_similarity * 100.0
        faith_pct = faith_norm * 100.0
        lang_pct  = lang_norm  * 100.0
        style_pct = style_norm * 100.0
        auto_fb = generate_feedback(sim_pct, faith_pct, lang_pct, style_pct)

        st.markdown("#### Ocena końcowa")
        g1, g2 = st.columns(2)
        with g1: st.metric("Wynik finalny ( % )", f"{final_pct:.0f}%")
        with g2: st.metric("Ocena (PL)", final_grade)

        st.markdown("#### Komentarz dla studenta")
        lead = "Świetny wynik." if final_pct >= 90 else "Dobry wynik." if final_pct >= 80 else \
               "Średni wynik." if final_pct >= 70 else "Wymaga pracy." if final_pct >= 60 else "Do gruntownej poprawy."
        teacher_comment = st.text_area("Uwagi prowadzącej (opcjonalnie)", height=100)
        final_comment = auto_fb if not teacher_comment.strip() else f"{auto_fb}\n\n**Uwagi prowadzącej:**\n{teacher_comment.strip()}"
        st.write(f"**{lead}**")
        st.write(final_comment)

        # ---------- ZAPIS WYNIKU DO TABELI (SESJA) ----------
        ensure_results_df()
        new_row = {
            "Data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Student": student_name or "—",
            "Zadanie/Plik": assignment_name or "—",
            "Podobieństwo_literalne": round(lit_sim, 3),
            "Podobieństwo_semantyczne": None if sem_sim is None else round(sem_sim, 3),
            "Wynik_łączny": round(combined_similarity, 3),
            "Wierność(1-5)": faithfulness, "Język(1-5)": language_quality, "Styl(1-5)": style,
            "W_auto": w_auto, "W_wierność": w_faith, "W_język": w_lang, "W_styl": w_style,
            "Progi(%): 5.0": th_5, "4.5": th_45, "4.0": th_40, "3.5": th_35, "3.0": th_30,
            "Wynik_finalny_%": round(final_pct, 1), "Ocena": final_grade,
        }
        st.session_state.results_df = pd.concat(
            [st.session_state.get("results_df", pd.DataFrame(columns=list(new_row.keys()))),
             pd.DataFrame([new_row])],
            ignore_index=True
        )

        # (Opcjonalnie) Zapis do Google Sheets
        if use_gsheets:
            if not GS_AVAILABLE:
                st.warning("Brak bibliotek Google Sheets. Sprawdź requirements.txt (gspread, google-auth).")
            else:
                try:
                    row_for_gs = [
                        new_row["Data"], new_row["Student"], new_row["Zadanie/Plik"],
                        new_row["Podobieństwo_literalne"], new_row["Podobieństwo_semantyczne"], new_row["Wynik_łączny"],
                        new_row["Wierność(1-5)"], new_row["Język(1-5)"], new_row["Styl(1-5)"],
                        new_row["W_auto"], new_row["W_wierność"], new_row["W_język"], new_row["W_styl"],
                        new_row["Progi(%): 5.0"], new_row["4.5"], new_row["4.0"], new_row["3.5"], new_row["3.0"],
                        new_row["Wynik_finalny_%"], new_row["Ocena"],
                    ]
                    gs_append_row(row_for_gs)
                    st.success("Zapisano wiersz do Google Sheets ✅")
                except Exception as e:
                    st.warning(f"Nie udało się zapisać do Google Sheets: {e}")

# ---------- WYNIKI ZBIORCZE ----------
ensure_results_df()
st.markdown("---")
st.subheader("📊 Zebrane wyniki (sesja)")

df = st.session_state.results_df.copy()

def _pct_col(series):
    return series.apply(lambda x: "" if pd.isna(x) else f"{float(x)*100:.0f}%")

if not df.empty:
    if "Podobieństwo_literalne" in df:   df["Podobieństwo_literalne"] = _pct_col(df["Podobieństwo_literalne"])
    if "Podobieństwo_semantyczne" in df: df["Podobieństwo_semantyczne"] = df["Podobieństwo_semantyczne"].apply(
        lambda x: "" if pd.isna(x) else f"{float(x)*100:.0f}%"
    )
    if "Wynik_łączny" in df:             df["Wynik_łączny"] = _pct_col(df["Wynik_łączny"])
    if "Wynik_finalny_%" in df:
        df["Wynik_finalny_%"] = df["Wynik_finalny_%"].apply(lambda x: "" if pd.isna(x) else f"{float(x):.0f}%")

st.dataframe(df, use_container_width=True)

if not st.session_state.results_df.empty:
    mean_pct = st.session_state.results_df["Wynik_finalny_%"].astype(float).mean()
    def _grade_to_float(g):
        try: return float(g)
        except: return np.nan
    mean_grade = st.session_state.results_df["Ocena"].apply(_grade_to_float).mean()

    st.markdown("### 📈 Średnie (sesja)")
    c1, c2 = st.columns(2)
    with c1: st.metric("Średni wynik ( % )", f"{mean_pct:.0f}%")
    with c2: st.metric("Średnia ocena (PL)", f"{mean_grade:.1f}")

# Pobieranie CSV
csv_data = st.session_state.results_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Pobierz wyniki jako CSV", data=csv_data, file_name="wyniki_tlumaczen.csv", mime="text/csv")
