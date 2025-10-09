import streamlit as st
import difflib
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ---------- USTAWIENIA STRONY ----------
st.set_page_config(page_title="Ocena tłumaczeń – wersja nauczycielska", layout="wide")

# ---------- MODEL SEMANTYCZNY ----------
@st.cache_resource
def load_st_model():
    # Lekki, szybki model do porównań zdań (ok. 22 MB)
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def best_semantic_match(student_text: str, refs: list[str]):
    """Zwraca (best_score (0..1), best_ref)."""
    model = load_st_model()
    emb_student = model.encode(student_text, normalize_embeddings=True)
    emb_refs = model.encode(refs, normalize_embeddings=True)
    sims = [float(np.dot(emb_student, r)) for r in emb_refs]  # dot == cosine (bo znormalizowane)
    idx = int(np.argmax(sims))
    return sims[idx], refs[idx]

# ---------- POMOCNICZE ----------
def compute_best_match(student_text: str, refs: list[str]):
    """Literalne podobieństwo; zwraca (best_score, best_ref, diff_preview_text)."""
    best_ref, best_score = "", 0.0
    for ref in refs:
        score = difflib.SequenceMatcher(None, student_text.lower(), ref.lower()).ratio()
        if score > best_score:
            best_score, best_ref = score, ref

    # Różnice — skrót, żeby było czytelnie
    diff_tokens = list(difflib.ndiff(student_text.split(), (best_ref or "").split()))
    if len(diff_tokens) > 200:
        diff_tokens = diff_tokens[:200] + ["..."]
    diff_preview = " ".join(diff_tokens)
    return best_score, best_ref, diff_preview

def auto_feedback(sim: float, faith: int, langq: int, style: int) -> str:
    """Krótki komentarz generowany na podstawie wyniku łącznego i rubryki."""
    tips = []
    if sim < 0.7:
        tips.append("Rozważ doprecyzowanie wierności względem sensu oryginału (parafrazy vs. znaczenie).")
    if faith <= 3:
        tips.append("Wzmocnij wierność przekładu: sprawdź kompletność informacji i unikanie nadinterpretacji.")
    if langq <= 3:
        tips.append("Popracuj nad poprawnością językową i kolokacjami w języku docelowym.")
    if style <= 3:
        tips.append("Zadbaj o rejestr i naturalność – płynność składni i dobór słownictwa.")
    return "Bardzo dobre tłumaczenie: wierne, poprawne i naturalne. Świetna robota!" if not tips else " ".join(tips)

def ensure_results_df():
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame(
            columns=[
                "Data", "Student", "Zadanie/Plik", "Podobieństwo_literalne",
                "Podobieństwo_semantyczne", "Wynik_łączny",
                "Wierność(1-5)", "Język(1-5)", "Styl(1-5)",
                "Komentarz", "Najbliższy_wzorzec", "Tłumaczenie_studenta"
            ]
        )

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### ⚙️ Instrukcja")
    st.markdown(
        "- Wklej **tekst źródłowy** (dla kontekstu – opcjonalnie), **tłumaczenie studenta** i **kilka wzorców**.\n"
        "- **Każde tłumaczenie wzorcowe oddziel pustą linią** (Enter, Enter).\n"
        "- Uzupełnij **rubrykę** i kliknij **Oceń tłumaczenie**.\n"
        "- Wyniki pojawią się poniżej; możesz je **pobrać jako CSV**."
    )
    st.caption("Uwaga: wynik łączny łączy semantykę i dopasowanie literalne.\n"
               "Dla elastycznych wariantów używaj kilku wzorców, by uniknąć krzywdzących ocen.")

st.title("📘 Ocena tłumaczeń studentów — wersja nauczycielska")

# ---------- FORMULARZ ----------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Dane wejściowe")
    student_name = st.text_input("👤 Imię i nazwisko studenta", "")
    assignment_name = st.text_input("🗂️ Nazwa zadania / pliku (opcjonalnie)", "")

    source_text = st.text_area("🧾 Tekst źródłowy (dla kontekstu, opcjonalnie)", height=120,
                               placeholder="Wklej oryginał (PL lub EN) — opcjonalnie.")
    student_translation = st.text_area("✍️ Tłumaczenie studenta", height=180,
                                       placeholder="Wklej tłumaczenie studenta, które chcesz ocenić.")

    reference_translations = st.text_area(
        "✅ Tłumaczenia wzorcowe (każde tłumaczenie oddziel **pustą linią**)",
        height=200,
        placeholder=(
            "Wklej jedno lub więcej poprawnych tłumaczeń.\n"
            "Każde tłumaczenie oddziel pustą linią (Enter, Enter).\n\n"
            "Przykład:\n"
            "He has lived in Warsaw for five years.\n\n"
            "He's been living in Warsaw for five years."
        )
    )

with col_right:
    st.subheader("Rubryka oceny")
    faithfulness = st.slider("Wierność (1–5)", 1, 5, 4)
    language_quality = st.slider("Poprawność językowa (1–5)", 1, 5, 4)
    style = st.slider("Styl / naturalność (1–5)", 1, 5, 4)

    st.subheader("Komentarz nauczyciela")
    teacher_comment = st.text_area("Wpisz własny komentarz (opcjonalnie)", height=120)

    st.subheader("Analiza")
    use_semantics = st.toggle(
        "Użyj analizy semantycznej (rekomendowane)",
        value=True,
        help="Porównuje znaczenie zdań (embeddingi), a nie tylko identyczność słów."
    )
    sem_weight = st.slider(
        "Waga semantyki w wyniku łącznym",
        0.0, 1.0, 0.7, 0.05,
        help="0 = tylko literalnie, 1 = tylko semantycznie"
    )

# ---------- PRZYCISK ----------
if st.button("🔎 Oceń tłumaczenie", type="primary"):
    if not student_translation.strip():
        st.error("Wprowadź tłumaczenie studenta.")
    elif not reference_translations.strip():
        st.error("Wprowadź co najmniej jedno tłumaczenie wzorcowe.")
    else:
        # --- LISTA WZORCÓW: rozdzielaj pustą linią (podwójny Enter) ---
        # Dzielimy po dwóch znakach nowej linii lub przynajmniej po pustym wierszu:
        raw = reference_translations.replace("\r\n", "\n")
        refs_list = [blk.strip() for blk in raw.split("\n\n") if blk.strip()]
        if len(refs_list) == 1 and "\n" in refs_list[0]:
            # Gdyby ktoś jednak wkleił po jednym Enterze – fallback do linii
            refs_list = [r.strip() for r in raw.split("\n") if r.strip()]

        # --- LITERALNE PODOBIEŃSTWO ---
        lit_sim, lit_best_ref, diff_preview = compute_best_match(student_translation, refs_list)  # 0..1

        # --- SEMANTYCZNE PODOBIEŃSTWO ---
        if use_semantics:
            sem_sim, sem_best_ref = best_semantic_match(student_translation, refs_list)  # 0..1
        else:
            sem_sim, sem_best_ref = None, None

        # --- WYBÓR WZORCA DO WYŚWIETLENIA ---
        display_best_ref = sem_best_ref if (use_semantics and sem_best_ref) else lit_best_ref

        # --- WYNIK ŁĄCZNY ---
        combined = (sem_weight * sem_sim + (1 - sem_weight) * lit_sim) if use_semantics else lit_sim

        # --- PREZENTACJA ---
        st.success("Analiza zakończona.")
        st.markdown("#### Najbliższe tłumaczenie wzorcowe")
        st.write(display_best_ref if display_best_ref else "—")

        met1, met2, met3 = st.columns(3)
        with met1:
            st.metric("Podobieństwo (literalne)", f"{lit_sim:.0%}")
        with met2:
            st.metric("Podobieństwo (semantyczne)", "—" if sem_sim is None else f"{sem_sim:.0%}")
        with met3:
            st.metric("Wynik łączny", f"{combined:.0%}")

        with st.expander("Podgląd różnic (skrót – wobec wzorca literalnego)"):
            st.code(diff_preview)

        # --- SZYBKIE „HINTY” (miękkie podpowiedzi) ---
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
            st.markdown("#### Potencjalne kwestie do sprawdzenia")
            for it in issues:
                st.write(f"- {it}")

        # --- KOMENTARZ (auto + Twój) ---
        auto_fb = auto_feedback(combined, faithfulness, language_quality, style)
        final_comment = auto_fb if not teacher_comment.strip() else f"{auto_fb}\n\n{teacher_comment.strip()}"

        st.markdown("#### Rubryka (Twoja ocena)")
        st.write(f"- **Wierność:** {faithfulness}/5")
        st.write(f"- **Poprawność językowa:** {language_quality}/5")
        st.write(f"- **Styl / naturalność:** {style}/5")
        st.markdown("#### Komentarz dla studenta")
        st.write(final_comment)

        # --- ZAPIS WYNIKU DO TABELI (SESJA) ---
        ensure_results_df()
        new_row = {
            "Data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Student": student_name or "—",
            "Zadanie/Plik": assignment_name or "—",
            "Podobieństwo_literalne": round(lit_sim, 3),
            "Podobieństwo_semantyczne": None if sem_sim is None else round(sem_sim, 3),
            "Wynik_łączny": round(combined, 3),
            "Wierność(1-5)": faithfulness,
            "Język(1-5)": language_quality,
            "Styl(1-5)": style,
            "Komentarz": final_comment,
            "Najbliższy_wzorzec": display_best_ref,
            "Tłumaczenie_studenta": student_translation,
        }
        st.session_state.results_df = pd.concat(
            [st.session_state.get("results_df", pd.DataFrame(columns=list(new_row.keys()))),
             pd.DataFrame([new_row])],
            ignore_index=True
        )

# ---------- WYNIKI ZBIORCZE ----------
ensure_results_df()
st.markdown("---")
st.subheader("📊 Zebrane wyniki (sesja)")
st.dataframe(st.session_state.results_df, use_container_width=True)

# Pobieranie CSV
csv_data = st.session_state.results_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Pobierz wyniki jako CSV", data=csv_data,
                   file_name="wyniki_tlumaczen.csv", mime="text/csv")
