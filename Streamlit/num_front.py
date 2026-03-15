import streamlit as st
import requests


st.title('MNIST Guess a number')

st.markdown('''Upload image of a :green[number]''')
st.markdown(':orange-badge[File must be .png | .jpg | .jpeg format]')
file = st.file_uploader('Choose file:', type=["jpg", "jpeg", "png"])
if file is not None:
    st.image(file, caption='Uploaded file',width=200)
    files = {
        'file': (file.name,file.getvalue(), file.type)
    }

if st.button('Check the Number'):
    if file is None:
        st.warning('Please upload a file')
        st.stop()
    try:
        response = requests.post('http://127.0.0.1:8000/numbers/predict/', files=files)
        if response.status_code == 200:
            prediction = response.json()
            st.success(f"Answer: {prediction.get('prediction')}")
        else:
            st.error(f"Server response: {response.status_code}")
    except requests.exceptions.RequestException:
        st.error("No connection!")

