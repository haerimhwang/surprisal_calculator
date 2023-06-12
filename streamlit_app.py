import streamlit as st
st.set_page_config(layout="wide")


import surprisal
import transformers

def load_css(file_name = "path/to/file.css"):
   with open(file_name) as f:
      st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css("style.css")


# Add css to make text bigger
st.markdown(
    """
    <style>

    textarea {
        font-size: 1.5rem !important;
    }
    input {
        font-size: 1.5rem !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)




######################
# Main page
######################

st.title('Surprisal calculator using GPT-2')

st.markdown(f'<p style="text-align: right; margin-bottom: 0"> Reference: <a href = "https://github.com/aalok-sathe/surprisal" target="_blank"> surprisal pacakge  </a> </p>', unsafe_allow_html=True)

st.markdown("""---""")

@st.cache_resource

def surprisal_calculator(input_text):

    from surprisal import model
    m = model.AutoTransformerModel.from_pretrained('gpt2')

    for result in m.surprise(input_text):
        return(result)


def plot_generator(input_text):
    from surprisal import model
    m = model.AutoTransformerModel.from_pretrained('gpt2')

    from matplotlib import pyplot as plt

    f, a = None, None

    for result in m.surprise(input_text):
        f, a = result.lineplot(f, a)

    #final_plot = plt.show()
    return f


def text_number_split(sentence_draft):
    head = sentence_draft.rstrip('0123456789')
    tail = sentence_draft[len(head):]
    return head, tail


sentence = st.text_input('Type your sentence below and press Enter/Return ⬇', 'Hi, there!', label_visibility="visible")

plot_output = plot_generator(sentence)

col1, col2, col3 = st.columns([0.4,0.1,0.5])

col1.header("surprisal value of each token")
sentence_draft = surprisal_calculator(sentence)
sentence_removed = str(sentence_draft).replace("Ġ", "")
sentence_output = text_number_split(sentence_removed)[0]
surprisal_output = text_number_split(sentence_removed)[1]

#col1.markdown('<p style="color: white; font-size: 20px;"> The surprisal value for this sentence is: </p>', unsafe_allow_html=True)
col1.markdown(f'<p style="color:#e8fc03; font-size: 20px;"> {sentence_output} </p>', unsafe_allow_html=True)
col1.markdown(f'<p style="color:#e8fc03; font-size: 20px;"> {surprisal_output} </p>', unsafe_allow_html=True)


col3.header("plot")
col3.pyplot(plot_output)





