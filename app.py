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
        placeholder=("Wklej jedno lub więcej
