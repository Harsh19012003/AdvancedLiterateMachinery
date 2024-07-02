from langchain.chains import SimpleSequentialChain, LLMChain
from langchain.prompts import PromptTemplate

codex = """You are an expert front-end developer. You develop beautiful webpages in HTML given an image description:

{img_desc}

Generate the HTML code with the right indentation.

===>Example start

<div class="container">
    <img src="img.jpg" alt="A beautiful image">
    <p>This is a beautiful image</p>
</div>

<===Example end

Begin!

"""
codex_template = PromptTemplate(input_variables=["img_desc"], template=codex)
codex_chain = LLMChain(llm=model, prompt=codex_template, output_key="story")