import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from images_duplicateurs import *

def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        "Prédiction avec dessin": png_export,
        "Membre du groupe":group_members
    }
    page = st.sidebar.selectbox(".", options=list(PAGES.keys()))
    PAGES[page]()

def png_export():
    image = st_canvas(update_streamlit=False, key="png_export",)
    if image is not None and image.image_data is not None:
        if st.button("Prediction", key ="prediction"):

            # TODO -- faire la fonction de prediction ici qui prend en paramètre une image et retoune une prediction.
            # orig_img = Image.open(image.image_data)
            # prediction(orig_img) qui retourne les proba

            st.write("perfect")
            st.balloons()
def group_members():
    st.write("## Composition du groupe")

    st.markdown("### Bourahima COULIBALY")
    st.markdown("### Tristan MARGATE")
    st.markdown("### Ahmed OSMAN")
    

if __name__ == "__main__":
    st.set_page_config(
        page_title="Deep learning project", page_icon=":pencil2:"
    )
    hide_streamlit_style = """            
                       <style>            
                                   
                       footer {visibility: hidden;}            
                       </style>            
                       """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("Deep learning project")
    st.sidebar.subheader("Les différentes pages")
    main()