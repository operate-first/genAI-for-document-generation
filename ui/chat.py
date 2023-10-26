#!/usr/bin/env python3
#

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


def user(message, history):
    return "", history + [[message, None]]


def bot(history):
    user_message = history[-1][0]
    new_user_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors="pt")

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([torch.LongTensor([]), new_user_input_ids], dim=-1)

    # generate a response
    response = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id).tolist()

    # convert the tokens to text, and then split the responses into lines
    response = tokenizer.decode(response[0]).split("<|endoftext|>")
    response = [(response[i], response[i + 1]) for i in range(0, len(response) - 1, 2)]  # convert to tuples of list
    history[-1] = response[0]
    return history


html = """
   <div class="star-rating" onclick="console.log('you clicked me');">
        <input type="radio" id="5-stars" name="rating" value="5"  onclick="ratingFn(5)" />
        <label for="5-stars" class="star">&#9733;</label>
        <input type="radio" id="4-stars" name="rating" value="4" onclick="ratingFn(4)" />
        <label for="4-stars" class="star">&#9733;</label>
        <input type="radio" id="3-stars" name="rating" value="3" onclick="ratingFn(3)" />
        <label for="3-stars" class="star">&#9733;</label>
        <input type="radio" id="2-stars" name="rating" value="2" onclick="ratingFn(2)" />
        <label for="2-stars" class="star">&#9733;</label>
        <input type="radio" id="1-star" name="rating" value="1"  onclick="ratingFn(1)" />
        <label for="1-star" class="star">&#9733;</label>
    </div>
"""

scripts = """
async () => {
    // set ratingFn() function on globalThis, so you html onlclick can access it
    globalThis.ratingFn = (val) => {
      document.getElementById('rating_text').innerHTML = "Rating is " + val;
    }
}
"""

css = """
/* component */

.star-rating {
  /*  border: solid 1px #ccc; */

    display: flex;
    flex-direction: row-reverse;
    font-size: 1.5em;
    justify-content: space-around;
    padding: 0 .2em;
    text-align: center;
    padding-left: 10px;
    width: 10em;
}

.star-rating input {
    display: none !important;
    border: 0px;
}

.star-rating label {
    color: #ccc;
    cursor: pointer;
}

.star-rating :checked~label {
    color: #f90;
}

.star-rating label:hover,
.star-rating label:hover~label {
    color: #fc0;
}


/* explanation */

article {
    background-color: #ffe;
    box-shadow: 0 0 1em 1px rgba(0, 0, 0, .25);
    color: #006;
    font-family: cursive;
    font-style: italic;
    margin: 4em;
    max-width: 30em;
    padding: 2em;
}
"""


def rating(rating):
    return f"Rating is, {rating}!"


with gr.Blocks(css=css) as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    radio_buttons = gr.HTML(html)
    rating_text = gr.Textbox(label="Rating", elem_id="rating_text")
    # radio_buttons.change(fn=rating, inputs=rating_text, outputs=rating_text)
    demo.load(None, None, None, _js=scripts)

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
