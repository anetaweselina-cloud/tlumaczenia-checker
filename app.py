import streamlit as st
import difflib
import pandas as pd
from datetime import datetime

# ---------- USTAWIENIA STRONY ----------
st.set_page_config(page_title="Ocena tłumaczeń – wersja nauczycielska", layout="wide")

# ---------- POMOCNICZE ----------
def compute_best_match(student_text: str, refs: list[str]):
    """Zwraca (best_score, best_ref, diff_preview_text)."""
    best_ref, best_score = "", 0.0
    for ref in refs:
        score = difflib.SequenceMatcher(None, student_text.lower(), ref.lower()).ratio()
        if score > best_score:
            best_score, best_ref = score, ref

    # Różnice (podgląd) – skrócone, żeby było czytelne
    diff_tokens = list(difflib.ndiff(student_text.split(), best_ref.split()))
    # utnij do 200 tokenów w podglądzie
    if len(diff_tokens) > 200:
        diff_tokens = diff_tokens[:200] + ["..."]
    diff_preview = " ".join(diff_tokens)

    return best_score, best_ref, diff_preview


def auto_feedback(sim: float, faith: int, langq: int, style: int) -> str:
    """Krótki komentarz generowany na podstawie podobieństwa i rubryki."""
    tips = []
    if sim < 0.7:
        tips.append("Rozważ doprecyzowanie wierności względem sensu oryginału (parafrazy vs. znaczenie).")
    if faith <= 3:
        tips.append("Wzmocnij wierność przekładu: sprawdź kompletność informacji i unikanie nadinterpretacji.")
    if langq <= 3:
        tips.append("Popracuj nad poprawnością językową i kolokacjami w języku docelowym.")
    if style <= 3:
        tips.append("Zadbaj o rejestr i naturalność – płynność składni i dobór słownictwa.")
    if not tips:
        return "Bardzo dobre tłumaczenie: wierne, poprawne i naturalne. Świetna robota!"
    return " ".join(tips)


def ensure_results_df():
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame(
            columns=[
                "Data", "Student", "Zadanie/Plik", "Podobieństwo",
                "Wierność(1-5)", "Język(1-5)", "Styl(1-5)",
                "Komentarz", "Najbliższy wzorzec", "Tłumaczenie studenta"
            ]
        )


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### ⚙️ Instrukcja")
    st.markdown(
        "- Wklej **tekst źródłowy** (dla kontekstu), **tłumaczenie studenta** i **kilka wzorców** (każdy w nowej linii).\n"
        "- Uzupełnij **rubrykę**.\n"
        "- Kliknij **Oceń tłumaczenie**.\n"
        "- Wyniki pojawią się poniżej i trafią do tabeli z możliwością **pobrania CSV**."
    )
    st.caption("Uwaga: podobieństwo jest literalne (tekstu do tekstu). "
               "Dla elastycznych wariantów używaj kilku wzorców, by uniknąć krzywdzących ocen.")

st.title("📘 Ocena tłumaczeń studentów — wersja nauczycielska")

# ---------- FORMULARZ DANYCH ----------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Dane wejściowe")
    student_name = st.text_input("👤 Imię i nazwisko studenta", "")
    assignment_name = st.text_input("🗂️ Nazwa zadania / pliku (opcjonalnie)", "")

    source_text = st.text_area("🧾 Tekst źródłowy (dla kontekstu, opcjonalnie)", height=120,
                               placeholder="Wklej oryginał (PL lub EN) — opcjonalnie.")
    student_translation = st.text_area("✍️ Tłumaczenie studenta", height=160,
                                       placeholder="Wklej tłumaczenie studenta, które chcesz ocenić.")
    reference_translations = st.text_area(
        "✅ Tłumaczenia wzorcowe (jedno wiersz = jeden wzorzec)",
        height=160,
        placeholder="Wklej kilka poprawnych tłumaczeń, każde w nowej linii.\n"
                    "Np.\nHe has lived in Warsaw for five years.\nHe’s been living in Warsaw for five years."
    )

with col_right:
    st.subheader("Rubryka oceny")
    faithfulness = st.slider("Wierność (1–5)", 1, 5, 4)
    language_quality = st.slider("Poprawność językowa (1–5)", 1, 5, 4)
    style = st.slider("Styl / naturalność (1–5)", 1, 5, 4)

    st.subheader("Komentarz nauczyciela")
    teacher_comment = st.text_area("Wpisz własny komentarz (opcjonalnie)", height=140)

# ---------- PRZYCISK OCENY ----------
if st.button("🔎 Oceń tłumaczenie", type="primary"):
    if not student_translation.strip():
        st.error("Wprowadź tłumaczenie studenta.")
    elif not reference_translations.strip():
        st.error("Wprowadź co najmniej jedno tłumaczenie wzorcowe.")
    else:
        refs_list = [r.strip() for r in reference_translations.split("\n") if r.strip()]
        sim, best_ref, diff_preview = compute_best_match(student_translation, refs_list)

        st.success("Analiza zakończona.")
        st.markdown("#### Najbliższe tłumaczenie wzorcowe")
        st.write(best_ref if best_ref else "—")
        st.metric("Podobieństwo (literalne)", f"{sim:.0%}")

        with st.expander("Podgląd różnic (skrót)"):
            st.code(diff_preview)

        # Komentarz automatyczny + zlepienie z komentarzem nauczyciela
        auto_fb = auto_feedback(sim, faithfulness, language_quality, style)
        final_comment = auto_fb if not teacher_comment.strip() else f"{auto_fb}\n\n{teacher_comment.strip()}"

        st.markdown("#### Rubryka (Twoja ocena)")
        st.write(f"- **Wierność:** {faithfulness}/5")
        st.write(f"- **Poprawność językowa:** {language_quality}/5")
        st.write(f"- **Styl / naturalność:** {style}/5")
        st.markdown("#### Komentarz dla studenta")
        st.write(final_comment)

        # Zapis do tabeli wyników (sesja)
        ensure_results_df()
        new_row = {
            "Data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Student": student_name or "—",
            "Zadanie/Plik": assignment_name or "—",
            "Podobieństwo": round(sim, 3),
            "Wierność(1-5)": faithfulness,
            "Język(1-5)": language_quality,
            "Styl(1-5)": style,
            "Komentarz": final_comment,
            "Najbliższy wzorzec": best_ref,
            "Tłumaczenie studenta": student_translation,
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
st.download_button("⬇️ Pobierz wyniki jako CSV", data=csv_data, file_name="wyniki_tlumaczen.csv", mime="text/csv")
