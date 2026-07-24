import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_searchbox import st_searchbox

st.set_page_config(page_title="Movie Blender", page_icon="🎬", layout="centered")

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w200"


@st.cache_resource
def load_artifacts():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def get_title_lookup(_movies_df):
    movies_df = _movies_df.drop_duplicates(subset="movieId").reset_index(drop=True)
    title_to_id = dict(zip(movies_df["title"], movies_df["movieId"]))
    id_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))
    all_titles = sorted(title_to_id.keys())
    return title_to_id, id_to_title, all_titles


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
qi = artifacts["qi"]  # item factor matrix, shape (n_items, n_factors)
raw_to_inner = artifacts["raw_to_inner"]
inner_to_raw = artifacts["inner_to_raw"]
title_to_id, id_to_title, all_titles = get_title_lookup(artifacts["movies_df"])

st.title("🎬 Movie Blender")
st.write(
    "Add movies, adjust how much each one should influence the result, "
    "and get recommendations blended from all of them."
)

if not st.secrets.get("TMDB_API_KEY"):
    st.caption("ℹ️ No TMDB_API_KEY found in secrets — posters will be skipped. See README for setup.")

if "selected" not in st.session_state:
    st.session_state.selected = []  # list of {"movieId": int, "title": str}


def weight_key(movie_id):
    return f"w_{movie_id}"


def slider_key(movie_id):
    return f"w_slider_{movie_id}"


def number_key(movie_id):
    return f"w_num_{movie_id}"


def set_weight(movie_id, value):
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
    # Note: we intentionally leave the w_/w_slider_/w_num_ keys for this movie in
    # session_state rather than deleting them. If we delete them here, an in-flight
    # slider/number_input update that was already en route from the browser can still
    # arrive after this and crash its on_change callback with a KeyError. Leaving the
    # orphaned keys around is harmless — they're just unused once the movie is removed.
    rebalance_weights()


def sync_from_slider(movie_id):
    if slider_key(movie_id) not in st.session_state:
        return
    val = st.session_state[slider_key(movie_id)]
    st.session_state[number_key(movie_id)] = val
    st.session_state[weight_key(movie_id)] = val


def sync_from_number(movie_id):
    if number_key(movie_id) not in st.session_state:
        return
    val = st.session_state[number_key(movie_id)]
    st.session_state[slider_key(movie_id)] = val
    st.session_state[weight_key(movie_id)] = val


# --- Search / add ---
# st_searchbox re-queries this function on every keystroke (debounced) and only ever
# sends back a handful of matches, so the dropdown updates live without lagging.
MAX_MATCHES = 25


def search_movies(searchterm: str):
    if not searchterm:
        return []
    q = searchterm.lower()
    starts = [t for t in all_titles if t.lower().startswith(q)]
    contains = [t for t in all_titles if q in t.lower() and t not in starts]
    return (starts + contains)[:MAX_MATCHES]


selected_title = st_searchbox(
    search_movies,
    key="movie_searchbox",
    placeholder="Start typing a title...",
    label="Search for a movie to add",
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
            set_weight(mid, 1.0)  # fallback, normally set by rebalance_weights

        poster_col, title_col, slider_col, num_col, remove_col = st.columns([1, 3, 3, 1.3, 0.6])

        with poster_col:
            poster_url = get_poster_url(mid)
            if poster_url:
                st.image(poster_url, width=60)
            else:
                st.write("🎬")

        with title_col:
            st.write(f"**{m['title']}**")

        with slider_col:
            st.slider(
                f"Weight slider — {m['title']}",
                min_value=0.0, max_value=1.0, step=0.01,
                key=slider_key(mid),
                on_change=sync_from_slider, args=(mid,),
                label_visibility="collapsed",
            )

        with num_col:
            st.number_input(
                f"Weight number — {m['title']}",
                min_value=0.0, max_value=1.0, step=0.01,
                key=number_key(mid),
                on_change=sync_from_number, args=(mid,),
                label_visibility="collapsed",
            )

        with remove_col:
            if st.button("✕", key=f"remove_{mid}"):
                remove_movie(mid)
                st.rerun()

    st.caption("Weights are relative — they don't need to add up to 1, they'll be normalized automatically. Use the slider or type an exact number.")

    n_recs = st.slider("Number of recommendations", min_value=5, max_value=25, value=10)

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
                # qi may now be a sparse matrix (hybrid SVD + content vectors) —
                # densify just this one row; qi as a whole stays sparse below.
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
                    for score, inner_iid in ranked:
                        raw_id = inner_to_raw[inner_iid]
                        title = id_to_title.get(raw_id, f"TMDB ID {raw_id}")
                        poster_url = get_poster_url(raw_id)

                        rec_poster_col, rec_text_col = st.columns([1, 5])
                        with rec_poster_col:
                            if poster_url:
                                st.image(poster_url, width=60)
                            else:
                                st.write("🎬")
                        with rec_text_col:
                            st.write(f"**{title}**")
                            st.caption(f"similarity: {score:.3f}")
