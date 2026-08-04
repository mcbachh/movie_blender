import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_searchbox import st_searchbox

st.set_page_config(page_title="Movie Blender", page_icon="🎬", layout="wide")

# Force columns to stay side-by-side at any screen width (mobile & desktop)
st.markdown(
    """
    <style>
    /* Google Font Imports for Cinema Vibe */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Montserrat:wght@500;700;800&display=swap');

    div[data-testid="stHorizontalBlock"] {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
    }
    div[data-testid="column"], div[data-testid="stColumn"] {
        min-width: 0 !important;
    }
    a, a:visited, a:hover, a:active {
        color: inherit !important;
        text-decoration: none !important;
    }
    a.about-link, a.about-link:visited {
        color: #ffd700 !important;
        text-decoration: underline !important;
        font-weight: bold;
    }
    
    /* -------------------------------------------------------------
       THEATER BACKGROUND: Rich Velvet Red Felt & Dark Vignette
       ------------------------------------------------------------- */
    .stApp {
        background: 
            radial-gradient(circle at center, rgba(80, 10, 15, 0.85) 0%, rgba(25, 4, 6, 0.98) 100%),
            repeating-radial-gradient(circle at 50% 50%, #3b0609, #3b0609 2px, #2d0406 2px, #2d0406 4px);
        background-color: #1a0204;
        font-family: 'Montserrat', sans-serif;
        color: #fce8c3;
    }

    /* -------------------------------------------------------------
       MARQUEE HEADER: Classic Theater Lightbulb Box
       ------------------------------------------------------------- */
    .marquee-header {
        position: relative;
        background: linear-gradient(180deg, #300003 0%, #150001 100%);
        border-radius: 20px;
        padding: 40px 20px;
        margin-bottom: 2rem;
        text-align: center;
        border: 4px solid #d4af37;
        box-shadow: 
            0 0 30px rgba(255, 195, 60, 0.4), 
            inset 0 0 40px rgba(0, 0, 0, 0.95);
    }

    /* Double Layer Outer Lightbulbs */
    .marquee-header::after {
        content: "";
        position: absolute;
        top: -16px;
        left: -16px;
        right: -16px;
        bottom: -16px;
        border-radius: 28px;
        pointer-events: none;
        
        /* Crisp Glowing Bulbs */
        background-image: radial-gradient(circle, #ffffff 0%, #fff099 20%, #ffaa00 45%, rgba(255,170,0,0) 70%);
        background-size: 26px 26px;
        background-position: top left;
        background-repeat: repeat;

        -webkit-mask: 
            linear-gradient(#fff 0 0) content-box, 
            linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        padding: 12px;
        
        filter: drop-shadow(0 0 8px #ffaa00) drop-shadow(0 0 16px #ff8800);
    }

    .marquee-title {
        font-family: 'Cinzel', serif;
        color: #ffe89e;
        font-size: 3.1rem;
        font-weight: 900;
        letter-spacing: 8px;
        text-transform: uppercase;
        text-shadow: 
            0 0 10px rgba(255, 232, 158, 0.8),
            0 0 25px rgba(212, 175, 55, 0.6),
            3px 3px 0px #000;
    }
    
    .marquee-subtitle {
        color: #e6c88b;
        font-size: 1.25rem;
        font-weight: 500;
        letter-spacing: 1px;
        margin-top: 8px;
        text-shadow: 1px 1px 2px #000;
    }

    /* -------------------------------------------------------------
       THEATER TABS: Brass & Velvet Buttons
       ------------------------------------------------------------- */
    div[data-testid="stTabs"] {
        margin-top: 1rem;
    }

    div[data-testid="stTabs"] button[role="tab"],
    button[data-baseweb="tab"] {
        background: linear-gradient(180deg, #4a0c10 0%, #290507 100%) !important;
        border: 2px solid #c9a227 !important;
        border-bottom: none !important;
        border-radius: 14px 14px 0 0 !important;
        height: 65px !important;
        padding: 0 40px !important;
        margin-right: 10px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 -4px 10px rgba(0,0,0,0.5) !important;
        transition: all 0.25s ease-in-out !important;
    }

    div[data-testid="stTabs"] button[role="tab"] *,
    button[data-baseweb="tab"] * {
        font-family: 'Cinzel', serif !important;
        font-size: 1.6rem !important;
        font-weight: 900 !important;
        color: #e6c88b !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        margin: 0 !important;
        padding: 0 !important;
        display: inline-block !important;
        visibility: visible !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.8) !important;
    }

    /* Hover Tab */
    div[data-testid="stTabs"] button[role="tab"]:hover,
    button[data-baseweb="tab"]:hover {
        background: linear-gradient(180deg, #6e1319 0%, #3d070b 100%) !important;
        border-color: #ffe27a !important;
    }

    div[data-testid="stTabs"] button[role="tab"]:hover *,
    button[data-baseweb="tab"]:hover * {
        color: #ffffff !important;
        text-shadow: 0 0 8px rgba(255,255,255,0.8) !important;
    }

    /* Active Selected Tab */
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(180deg, #8c171f 0%, #520b10 100%) !important;
        border: 2px solid #ffd700 !important;
        border-bottom: 5px solid #ffd700 !important;
        box-shadow: 0 -4px 20px rgba(255, 215, 0, 0.4) !important;
    }

    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] *,
    button[data-baseweb="tab"][aria-selected="true"] * {
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.7) !important;
    }

    /* Remove default underline bar */
    div[data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"] {
        display: none !important;
    }

    /* -------------------------------------------------------------
       CONTAINERS & CARDS: Brass Framing & Ticket Aesthetics
       ------------------------------------------------------------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(180deg, rgba(42, 8, 11, 0.95) 0%, rgba(20, 3, 5, 0.95) 100%) !important;
        border: 2px solid #b9973d !important;
        border-radius: 14px !important;
        padding: 1.5rem !important;
        box-shadow: inset 0 0 15px rgba(0,0,0,0.8), 0 4px 15px rgba(0,0,0,0.6) !important;
    }

    /* -------------------------------------------------------------
       SPOTLIGHT PEDESTAL (#1 REC FEATURE)
       ------------------------------------------------------------- */
    .pedestal-container {
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 2.5rem auto 3rem auto;
        padding: 40px 20px 30px 20px;
        background: radial-gradient(circle at center, rgba(80, 12, 18, 0.95) 0%, rgba(25, 4, 6, 0.98) 80%);
        border: 3px solid #ffd700;
        border-radius: 20px;
        box-shadow: 0 0 35px rgba(255, 215, 0, 0.35), inset 0 0 40px rgba(0, 0, 0, 0.9);
        overflow: hidden;
        text-align: center;
    }

    /* Animated Pinwheel Background Lightburst */
    .pinwheel-bg {
        position: absolute;
        top: 30%;
        left: 50%;
        width: 500px;
        height: 500px;
        margin-top: -250px;
        margin-left: -250px;
        border-radius: 50%;
        background: repeating-conic-gradient(
            from 0deg,
            rgba(255, 215, 0, 0.15) 0deg 15deg,
            transparent 15deg 30deg
        );
        animation: spin 20s linear infinite;
        pointer-events: none;
        z-index: 1;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Poster Wrapper on Pedestal */
    .pedestal-poster-wrap {
        position: relative;
        z-index: 2;
        margin-bottom: 20px;
    }

    .pedestal-poster-img {
        width: 170px;
        border: 4px solid #ffd700;
        border-radius: 10px;
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.6), 0 10px 25px rgba(0,0,0,0.9);
        transition: transform 0.3s ease;
    }

    .pedestal-poster-img:hover {
        transform: scale(1.04);
    }

    .pedestal-details {
        position: relative;
        z-index: 2;
    }

    .pedestal-badge {
        font-family: 'Cinzel', serif;
        background: linear-gradient(180deg, #ffd700 0%, #b8860b 100%);
        color: #ffffff;
        font-weight: 900;
        font-size: 1.1rem;
        padding: 4px 18px;
        border: 2px solid #ffe89e !important;
        border-radius: 8px !important;
        letter-spacing: 2px;
        text-transform: uppercase;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.8);
        text-shadow: 0px 1px 3px rgba(0, 0, 0, 0.8) !important;
        display: inline-block;
        margin-bottom: 12px;
    }

    .pedestal-title {
        font-family: 'Cinzel', serif;
        font-size: 2.2rem;
        font-weight: 900;
        color: #ffffff;
        text-shadow: 0 0 12px rgba(255, 232, 158, 0.9), 2px 2px 4px #000;
        margin-bottom: 6px;
    }

    .pedestal-similarity {
        color: #ffd700;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 1px;
    }

    /* -------------------------------------------------------------
       NUMBERED RUNNER-UP LIST (#2+)
       ------------------------------------------------------------- */
    .rec-item-card {
        background: linear-gradient(180deg, rgba(35, 6, 9, 0.8) 0%, rgba(18, 3, 5, 0.9) 100%);
        border: 1px solid #b9973d;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }

    .rec-number-badge {
        font-family: 'Cinzel', serif;
        background: linear-gradient(180deg, #520b10 0%, #290507 100%);
        border: 2px solid #d4af37;
        color: #ffe27a;
        font-size: 0.8rem;
        font-weight: 900;
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 8px rgba(212, 175, 55, 0.3);
        flex-shrink: 0;
    }

    /* -------------------------------------------------------------
       BUTTONS: Brass / Gold Vintage Buttons with White Text
       ------------------------------------------------------------- */
    .stButton button {
        background: linear-gradient(180deg, #d4af37 0%, #8a6d14 100%) !important;
        border: 2px solid #ffe89e !important;
        border-radius: 8px !important;
        box-shadow: 0 3px 8px rgba(0,0,0,0.5) !important;
        transition: all 0.2s ease-in-out !important;
        padding: 0.5rem 1.5rem !important;
    }

    .stButton button * {
        font-family: 'Montserrat', sans-serif !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        text-shadow: 0px 1px 3px rgba(0, 0, 0, 0.8) !important;
    }

    .stButton button:hover {
        background: linear-gradient(180deg, #edd068 0%, #a88720 100%) !important;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.7) !important;
        transform: translateY(-1px);
    }

    .stButton button:active {
        transform: translateY(1px);
    }

    /* Primary Recommendation Button Glow */
    .stButton button[kind="primary"] {
        background: linear-gradient(180deg, #b81d24 0%, #630c10 100%) !important;
        border: 2px solid #ffd700 !important;
    }
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(180deg, #d9252e 0%, #821217 100%) !important;
        box-shadow: 0 0 20px rgba(255, 0, 0, 0.6), 0 0 10px rgba(255, 215, 0, 0.8) !important;
    }

    /* Headers typography */
    h1, h2, h3 {
        font-family: 'Cinzel', serif !important;
        color: #ffe89e !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8) !important;
    }
    
    /* Input Searchbox overrides */
    div[data-baseweb="input"] {
        background-color: #1a0406 !important;
        border: 1px solid #c9a227 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w300"


def tmdb_movie_url(movie_id):
    return f"https://www.themoviedb.org/movie/{movie_id}"


@st.cache_resource
def load_artifacts():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def get_title_lookup(_movies_df):
    movies_df = _movies_df.drop_duplicates(subset="movieId").reset_index(drop=True)
    label_to_id = {}
    id_to_label = {}

    for _, row in movies_df.iterrows():
        year = row.get("release_year")
        if pd.isna(year) or year in (None, "", "None"):
            label = str(row["title"])
        else:
            label = f"{row['title']} ({int(float(year))})"

        if label in label_to_id:
            label = f"{label} [id:{row['movieId']}]"

        label_to_id[label] = row["movieId"]
        id_to_label[row["movieId"]] = label

    all_labels = sorted(label_to_id.keys())
    return label_to_id, id_to_label, all_labels


@st.cache_data(show_spinner=False)
def get_poster_url(tmdb_id):
    api_key = st.secrets.get("TMDB_API_KEY")
    if not api_key:
        return None
    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": api_key},
            timeout=5,
        )
        if resp.status_code == 200:
            poster_path = resp.json().get("poster_path")
            if poster_path:
                return f"{TMDB_IMAGE_BASE}{poster_path}"
    except requests.RequestException:
        pass
    return None


artifacts = load_artifacts()
qi = artifacts["qi"]
raw_to_inner = artifacts["raw_to_inner"]
inner_to_raw = artifacts["inner_to_raw"]

covered_ids = set(raw_to_inner.keys())
movies_df_covered = artifacts["movies_df"][artifacts["movies_df"]["movieId"].isin(covered_ids)]

title_to_id, id_to_title, all_titles = get_title_lookup(movies_df_covered)

if "selected" not in st.session_state:
    st.session_state.selected = []


def weight_key(movie_id):
    return f"w_{movie_id}"


def slider_key(movie_id):
    return f"w_slider_{movie_id}"


def number_key(movie_id):
    return f"w_num_{movie_id}"


def set_weight(movie_id, value):
    value = round(value, 4)
    st.session_state[weight_key(movie_id)] = value
    st.session_state[slider_key(movie_id)] = value
    st.session_state[number_key(movie_id)] = value


def rebalance_weights():
    n = len(st.session_state.selected)
    if n == 0:
        return
    even = round(1.0 / n, 4)
    for m in st.session_state.selected:
        set_weight(m["movieId"], even)


def redistribute_others(changed_id, new_value):
    new_value = max(0.0, min(1.0, new_value))
    others = [m for m in st.session_state.selected if m["movieId"] != changed_id]

    if not others:
        set_weight(changed_id, 1.0)
        return

    remaining = 1.0 - new_value
    others_total = sum(st.session_state.get(weight_key(m["movieId"]), 0.0) for m in others)

    if others_total <= 0:
        even = remaining / len(others)
        for m in others:
            set_weight(m["movieId"], even)
    else:
        for m in others:
            old = st.session_state.get(weight_key(m["movieId"]), 0.0)
            set_weight(m["movieId"], remaining * (old / others_total))

    set_weight(changed_id, new_value)


def add_movie(title):
    if not title:
        return
    movie_id = title_to_id[title]
    if any(m["movieId"] == movie_id for m in st.session_state.selected):
        st.toast(f"{title} is already added.")
        return
    st.session_state.selected.append({"movieId": movie_id, "title": title})
    rebalance_weights()


def remove_movie(movie_id):
    st.session_state.selected = [m for m in st.session_state.selected if m["movieId"] != movie_id]
    rebalance_weights()


def sync_from_slider(movie_id):
    if slider_key(movie_id) not in st.session_state:
        return
    redistribute_others(movie_id, st.session_state[slider_key(movie_id)])


def sync_from_number(movie_id):
    if number_key(movie_id) not in st.session_state:
        return
    redistribute_others(movie_id, st.session_state[number_key(movie_id)])


MAX_MATCHES = 25


def search_movies(searchterm: str):
    if not searchterm:
        return []
    q = searchterm.lower()
    starts = [t for t in all_titles if t.lower().startswith(q)]
    contains = [t for t in all_titles if q in t.lower() and t not in starts]
    return (starts + contains)[:MAX_MATCHES]


st.markdown(
    """<div class="marquee-header">
        <div class="marquee-title">MOVIE BLENDER</div>
        <div class="marquee-subtitle">Blend movies into a list of perfect recommendations</div>
    </div>""",
    unsafe_allow_html=True,
)

# =========================================================================
# Tabs
# =========================================================================
tab_blender, tab_about = st.tabs(["Movie Blender", "About"])

with tab_blender:
    st.write(
        "Everyone wants to watch something different on movie night? "
        "This website allows you to get movie recommendations from multiple input movies, making picking what to watch so much easier! "
        "Select at least one movie to get started. \n \n" 
        "(NOTE: Due to limited data, only movies from 2023 and before are available. "
        "Movies released during 2023 may have low quality recommendations because of poor rating coverage.)"
    )

    if not st.secrets.get("TMDB_API_KEY"):
        st.caption("ℹ️ No TMDB_API_KEY found in secrets — posters will be skipped. See README for setup.")

    selected_title = st_searchbox(
        search_movies,
        key="movie_searchbox",
        placeholder="Start typing a title...",
    )

    if st.button("Add movie", disabled=not selected_title):
        add_movie(selected_title)
        st.session_state.pop("movie_searchbox", None)
        st.rerun()

    st.divider()

    # --- Tiles ---
    if not st.session_state.selected:
        st.info("Add at least one movie to get started.")
    else:
        st.subheader("Your movies")
        for m in st.session_state.selected:
            mid = m["movieId"]
            if weight_key(mid) not in st.session_state:
                set_weight(mid, 1.0)

            poster_col, content_col = st.columns([1, 4])

            with poster_col:
                poster_url = get_poster_url(mid)
                url = tmdb_movie_url(mid)
                if poster_url:
                    st.markdown(
                        f'<a href="{url}" target="_blank"><img src="{poster_url}" width="65" style="border: 2px solid #d4af37; border-radius: 4px; box-shadow: 0 4px 8px rgba(0,0,0,0.8);"></a>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f'<a href="{url}" target="_blank">[POSTER MISSING]</a>', unsafe_allow_html=True)

            with content_col:
                st.markdown(f"**[{m['title']}]({tmdb_movie_url(mid)})**")
                slider_col, num_col, remove_col = st.columns([3, 1.3, 0.6])

                with slider_col:
                    st.slider(
                        f"Weight slider — {m['title']}",
                        min_value=0.0, max_value=1.0, step=0.01,
                        key=slider_key(mid),
                        on_change=sync_from_slider, args=(mid,),
                        label_visibility="collapsed",
                    )
                    st.caption("Adjust how much this movie will impact the recommendations.")

                with num_col:
                    st.number_input(
                        f"Weight number — {m['title']}",
                        min_value=0.0, max_value=1.0, step=0.01,
                        key=number_key(mid),
                        on_change=sync_from_number, args=(mid,),
                        label_visibility="collapsed",
                    )

                with remove_col:
                    st.button("✕", key=f"remove_{mid}", on_click=remove_movie, args=(mid,))

        n_recs = st.slider("Number of recommendations", min_value=1, max_value=25, value=1)

        if st.button("Get recommendations", type="primary"):
            weights_raw = [st.session_state.get(weight_key(m["movieId"]), 1.0) for m in st.session_state.selected]
            total_weight = sum(weights_raw)

            if total_weight == 0:
                st.error("At least one movie needs a non-zero weight.")
            else:
                input_inner_iids = []
                vectors = []
                weights = []
                skipped = []

                for m, w in zip(st.session_state.selected, weights_raw):
                    raw_id = m["movieId"]
                    if raw_id not in raw_to_inner:
                        skipped.append(m["title"])
                        continue
                    inner_iid = raw_to_inner[raw_id]
                    input_inner_iids.append(inner_iid)
                    row = qi[inner_iid]
                    if hasattr(row, "toarray"):
                        row = row.toarray().ravel()
                    vectors.append(row)
                    weights.append(w / total_weight)

                if skipped:
                    st.warning(f"Skipped (not enough ratings in training data): {', '.join(skipped)}")

                if not vectors:
                    st.error("None of the selected movies have enough rating data to recommend from.")
                else:
                    vectors = np.array(vectors)
                    weights = np.array(weights).reshape(-1, 1)
                    weighted_avg = np.sum(vectors * weights, axis=0, keepdims=True)

                    sims = cosine_similarity(weighted_avg, qi)[0]
                    input_set = set(input_inner_iids)

                    ranked = sorted(
                        (
                            (score, inner_iid)
                            for inner_iid, score in enumerate(sims)
                            if inner_iid not in input_set
                        ),
                        key=lambda x: x[0],
                        reverse=True,
                    )[:n_recs]

                    st.subheader("Recommendations")
                    with st.spinner("Loading posters..."):
                        if ranked:
                            # -------------------------------------------------------------
                            # #1 RECOMMENDATION SPOTLIGHT PEDESTAL
                            # -------------------------------------------------------------
                            top_score, top_inner_iid = ranked[0]
                            top_raw_id = inner_to_raw[top_inner_iid]
                            top_title = id_to_title.get(top_raw_id, f"TMDB ID {top_raw_id}")
                            top_poster_url = get_poster_url(top_raw_id)
                            top_url = tmdb_movie_url(top_raw_id)

                            poster_html = (
                                f'<a href="{top_url}" target="_blank"><img class="pedestal-poster-img" src="{top_poster_url}"></a>'
                                if top_poster_url
                                else f'<a href="{top_url}" target="_blank" style="color:#ffe89e;font-size:1.2rem;">[POSTER MISSING]</a>'
                            )

                            pedestal_html = f"""
                            <div class="pedestal-container">
                                <div class="pinwheel-bg"></div>
                                <div class="pedestal-poster-wrap">
                                    {poster_html}
                                </div>
                                <div class="pedestal-details">
                                    <div class="pedestal-badge">#1 TOP MATCH</div>
                                    <div class="pedestal-title"><a href="{top_url}" target="_blank">{top_title}</a></div>
                                    <div class="pedestal-similarity">Similarity Score: {top_score:.3f}</div>
                                </div>
                            </div>
                            """
                            st.markdown(pedestal_html, unsafe_allow_html=True)

                            # -------------------------------------------------------------
                            # RUNNER-UP RECOMMENDATIONS (#2+)
                            # -------------------------------------------------------------
                            if len(ranked) > 1:
                                st.markdown("### Other Selections")
                                for rank_idx, (score, inner_iid) in enumerate(ranked[1:], start=2):
                                    raw_id = inner_to_raw[inner_iid]
                                    title = id_to_title.get(raw_id, f"TMDB ID {raw_id}")
                                    poster_url = get_poster_url(raw_id)
                                    url = tmdb_movie_url(raw_id)

                                    rec_badge_col, rec_poster_col, rec_text_col = st.columns([0.6, 1, 5])
                                    
                                    with rec_badge_col:
                                        st.markdown(f'<div class="rec-number-badge">#{rank_idx}</div>', unsafe_allow_html=True)

                                    with rec_poster_col:
                                        if poster_url:
                                            st.markdown(
                                                f'<a href="{url}" target="_blank"><img src="{poster_url}" width="65" style="border: 2px solid #d4af37; border-radius: 4px; box-shadow: 0 4px 8px rgba(0,0,0,0.8);"></a>',
                                                unsafe_allow_html=True,
                                            )
                                        else:
                                            st.markdown(f'<a href="{url}" target="_blank">[POSTER MISSING]</a>', unsafe_allow_html=True)

                                    with rec_text_col:
                                        st.markdown(f"**[{title}]({url})**")
                                        st.caption(f"similarity: {score:.3f}")

with tab_about:
    st.title("About")

    ABOUT_TEXT = """
**THE IDEA:** \n   
This website was created in summer 2026 as the culmination of a project that I started to get introduced
 to recommendation systems. I'm a big movie fan and wanted to create a model that could give decent movie recommendations
 based on an input movie to help me expand my watchlist. After experimenting with different approaches, I found that
 combining collaborative filtering with content-based features gave the best results for generating recommendations that
 are both relevant and unique. At that point, I thought of an idea I had come up with years ago for a system that could blend two songs together
 to recommend new songs that are similar to a combination of the inputs. I realized I could apply the same idea to movies, blending multiple 
 input movies to generate recommendations that reflect a combination of the selected movies. This led to the creation of the Movie Blender website.

----
**HOW IT WORKS:** \n
The basic concept behind the Movie Blender is that it averages vector embeddings of the input movies to create a combined vector in space,
 and then finds movies closest to that combined vector, effectively recommending movies that are similar to the combination of the input movies.
 The embeddings consider factors like genre, director, keywords, and popularity, among other features. The vector also includes the embeddings found
from a 50-factor collaborative filtering model trained on historical user ratings, which captures hidden relationships between movies.

----
**CITATIONS:** \n
- Ratings Dataset: <a class="about-link" href="https://grouplens.org/datasets/movielens/latest/" target="_blank">MovieLens Latest Full Dataset</a>
- Movie Data: <a class="about-link" href="https://www.themoviedb.org/" target="_blank">TMDB</a>
    - (This product uses the TMDB API but is not endorsed or certified by TMDB.)
----
**ADDITIONAL SOURCES:** \n
- GitHub: <a class="about-link" href="https://github.com/mcbachh/movie_blender" target="_blank">mcbachh/movie_blender</a>

---- 
**ABOUT ME:** \n
This project was created by Michael Bachman, Data Science, Tufts '28 \n
You can find more of my work on <a class="about-link" href="https://github.com/mcbachh" target="_blank">GitHub</a>. \n
Connect on <a class="about-link" href="https://www.linkedin.com/in/michael-bachman-60655a344/" target="_blank">LinkedIn</a>. \n
And follow me on Letterboxd (Username: mcbach) :)
"""

    with st.container(border=True):
        st.markdown(ABOUT_TEXT, unsafe_allow_html=True)