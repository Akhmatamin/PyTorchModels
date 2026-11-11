import streamlit as st
import requests

API_URL = 'http://127.0.0.1:8000'

st.set_page_config(
    page_title='ML Models Dashboard',
    page_icon='🧠',
    layout='wide',
)

st.markdown('''
<style>
.hero {
    padding: 2.5rem 2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 50%, #00c9a7 100%);
    background-size: 200% 200%;
    animation: shift 10s ease infinite;
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(37, 117, 252, 0.3);
}
.hero h1 { color: white; font-size: 2.6rem; margin: 0; padding: 0; }
.hero p { font-size: 1.15rem; opacity: 0.92; margin: 0.6rem 0 0 0; }
@keyframes shift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.card-emoji { font-size: 2.4rem; line-height: 1; }
.card-title { font-size: 1.2rem; font-weight: 700; margin: 0.5rem 0 0.2rem 0; }
.card-text { opacity: 0.75; font-size: 0.92rem; min-height: 2.8rem; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(106, 17, 203, 0.18);
}
.footer { text-align: center; opacity: 0.6; font-size: 0.85rem; margin-top: 2rem; }
</style>
''', unsafe_allow_html=True)


MODELS = [
    # (file, title, nav icon, emoji, description, section)
    ('num_front.py', 'Guess number', ':material/pin:', '🔢',
     'Handwritten digit recognition trained on MNIST.', 'Images'),
    ('fashion_front.py', 'Guess fashion', ':material/checkroom:', '👕',
     'Clothing classification with Fashion-MNIST.', 'Images'),
    ('cifar10_front.py', 'Guess cifar10', ':material/image:', '🖼️',
     'Recognize 10 everyday object classes.', 'Images'),
    ('cifar100_front.py', 'Guess cifar100', ':material/photo_library:', '🎨',
     'Fine-grained recognition across 100 classes.', 'Images'),
    ('transport_front.py', 'Guess transport', ':material/directions_car:', '🚗',
     'Airplanes, buses, F1 cars, motorbikes and sedans.', 'Images'),
    ('flower_front.py', 'Guess flower', ':material/local_florist:', '🌸',
     'Tell daisies, roses and tulips apart.', 'Images'),
    ('image_scene_front.py', 'Guess image', ':material/landscape:', '🏞️',
     'Scene classification: forest, sea, city...', 'Images'),
    ('trash_front.py', 'Guess garbage', ':material/recycling:', '♻️',
     'Sort waste into the right recycling bin.', 'Images'),
    ('audio_front.py', 'Guess audio', ':material/graphic_eq:', '🎧',
     'Recognize 35 spoken command words.', 'Audio'),
]

model_pages = {
    file: st.Page(file, title=title, icon=icon)
    for file, title, icon, *_ in MODELS
}


@st.cache_data(ttl=10, show_spinner=False)
def api_is_online():
    try:
        return requests.get(f'{API_URL}/docs', timeout=1).status_code == 200
    except requests.exceptions.RequestException:
        return False


def home():
    st.markdown('''
    <div class="hero">
        <h1>🧠 ML Models Dashboard</h1>
        <p>Upload an image or audio clip and let a PyTorch model guess what it is ✨</p>
    </div>
    ''', unsafe_allow_html=True)

    image_count = sum(1 for m in MODELS if m[5] == 'Images')
    audio_count = len(MODELS) - image_count
    c1, c2, c3 = st.columns(3)
    c1.metric('🤖 Models', len(MODELS))
    c2.metric('🖼️ Image models', image_count)
    c3.metric('🎧 Audio models', audio_count)

    st.subheader('Pick a model to try')
    cols = st.columns(3)
    for i, (file, title, _, emoji, description, _) in enumerate(MODELS):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f'''
                <div class="card-emoji">{emoji}</div>
                <div class="card-title">{title}</div>
                <div class="card-text">{description}</div>
                ''', unsafe_allow_html=True)
                st.page_link(model_pages[file], label='Try it', icon=':material/arrow_forward:')

    with st.expander('💡 How it works'):
        st.markdown('''
        1. **Choose** a model from the cards above or the sidebar.
        2. **Upload** a file (image or audio).
        3. **Predict** — the file is sent to the FastAPI backend and the model returns its guess.
        ''')

    st.markdown('<div class="footer">Built with PyTorch · FastAPI · Streamlit</div>',
                unsafe_allow_html=True)


sections = {'': [st.Page(home, title='Home', icon=':material/home:', default=True)]}
for file, _, _, _, _, section in MODELS:
    sections.setdefault(section, []).append(model_pages[file])

pages = st.navigation(sections)

with st.sidebar:
    st.markdown('### 🧠 ML Dashboard')
    if api_is_online():
        st.success('API online', icon=':material/check_circle:')
    else:
        st.error('API offline — run `python main.py`', icon=':material/error:')

pages.run()
