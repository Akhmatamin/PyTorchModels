import streamlit as st

pages = st.navigation([
    st.Page('num_front.py', title='Guess number'),
    st.Page('fashion_front.py', title='Guess fashion'),
    st.Page('cifar10_front.py', title='Guess cifar10'),
    st.Page('cifar100_front.py', title='Guess cifar100'),
    st.Page('transport_front.py', title='Guess transport'),
    st.Page('flower_front.py', title='Guess flower'),
])

pages.run()