In last section, we just put a RAG process by assembly all components together. We can gain first hand experience for RAG process but we don't know the logic behide it, that is why and how each component come into its place, why we need that
component and why we need to assembly then in the given way. In this section, we will begin from a raw project, and check what kind of problems or diffculties we will have and see why we need the given component to save life, but doing this,
we will know the logic behide the whole RAG process.

We still using the notebook from previous section, ensure the given packages and api key configurations. In last section, we using a prompt component from langchain to help us create a prompt, this time we first try to do it by our own.
Let's add the following code to generate prompt by ourself:

```py
from openai import OpenAI
import time
client = OpenAI()
gptmodel = "gpt-4o"

def generate_answer(prompt):
  print(f"generate answer prompt: {prompt}")
  try:
    start = time.time()
    response = client.chat.completions.create(
        model = gptmodel,
        messages = [
            {'role':"system", "content": "You are an expert of llm and RAG"},
            {'role':"assistant", "content": "You can explain read the input and answer in detail"},
            {'role': 'user', 'content': prompt}
        ],
        temperature = 0.1, #let the model has some kind of freedom to imaginate
    )
    #this is the job of StrOutputParser
    return response.choices[0].message.content.strip()
  except Exception as e:
    return str(e)

def call_llm_with_self_prompot(questions):
  #text_input = '\n'.join(questions)
  #we do this by langchain hub
  prompt = f"Please elaborate on the following content:\n {questions}"
  print(f"prompt is : {prompt}")
  return generate_answer(prompt)

import textwrap
#this is the job of StrOutputParser
def print_formatted_response(response):
  wrapper = textwrap.TextWrapper(width = 80) #80 words in a line
  wrapped_text = wrapper.fill(text=response)
  print("Response:")
  print("----------------")
  print(wrapped_text)
  print("----------------")
```
Run the code aboved and we can call the aboved call as following:
```py
query = "tell me something about rag store"
llm_response = call_llm_with_self_prompot(query)
response = print_formatted_response(llm_response)
print(response)
```
Then we get the following answer:
```py
Retrieval Augmented Generation (RAG) represents a sophisticated hybrid approach
in the field of artificial intelligence, particularly within the realm of
natural language processing (NLP). It innovatively combines the capabilities of
neural network-based language models with retrieval systems to enhance the
generation of text, making it more accurate, informative, and contextually
relevant. This methodology leverages the strengths of both generative and
retrieval architectures to tackle complex tasks that require not only linguistic
fluency but also factual correctness and depth of knowledge. At the core of
Retrieval Augmented Generation (RAG) is a generative model, typically a
transformer-based neural network, similar to those used in models like GPT
(Generative Pre-trained Transformer) or BERT (Bidirectional Encoder
Representations from Transformers). This component is responsible for producing
coherent and contextually appropriate language outputs based on a mixture of
input prompts and additional information fetched by the retrieval component.
Complementing the language model is the retrieval system, which is usually built
on a database of documents or a corpus of texts. This system uses techniques
from information retrieval to find and fetch documents that are relevant to the
input query or prompt. The mechanism of relevance determination can range from
simple keyword matching to more complex semantic search algorithms which
interpret the meaning behind the query to find the best matches. This component
merges the outputs from the language model and the retrieval system. It
effectively synthesizes the raw data fetched by the retrieval system into the
generative process of the language model. The integrator ensures that the
information from the retrieval system is seamlessly incorporated into the final
text output, enhancing the model's ability to generate responses that are not
only fluent and grammatically correct but also rich in factual details and
context-specific nuances. When a query or prompt is received, the system first
processes it to understand the requirement or the context. Based on the
processed query, the retrieval system searches through its database to find
relevant documents or information snippets. This retrieval is guided by the
similarity of content in the documents to the query, which can be determined
through various techniques like vector embeddings or semantic similarity
measures. The retrieved documents are then fed into the language model. In some
implementations, this integration happens at the token level, where the model
can access and incorporate specific pieces of information from the retrieved
texts dynamically as it generates each part of the response. The language model,
now augmented with direct access to retrieved information, generates a response.
This response is not only influenced by the training of the model but also by
the specific facts and details contained in the retrieved documents, making it
more tailored and accurate. By directly incorporating information from external
sources, Retrieval Augmented Generation (RAG) models can produce responses that
are more factual and relevant to the given query. This is particularly useful in
domains like medical advice, technical support, and other areas where precision
and up-to-date knowledge are crucial. Retrieval Augmented Generation (RAG)
systems can dynamically adapt to new information since they retrieve data in
real-time from their databases. This allows them to remain current with the
latest knowledge and trends without needing frequent retraining. With access to
a wide range of documents, Retrieval Augmented Generation (RAG) systems can
provide detailed and nuanced answers that a standalone language model might not
be capable of generating based solely on its pre-trained knowledge. While
Retrieval Augmented Generation (RAG) offers substantial benefits, it also comes
with its challenges. These include the complexity of integrating retrieval and
generation systems, the computational overhead associated with real-time data
retrieval, and the need for maintaining a large, up-to-date, and high-quality
database of retrievable texts. Furthermore, ensuring the relevance and accuracy
of the retrieved information remains a significant challenge, as does managing
the potential for introducing biases or errors from the external sources. In
summary, Retrieval Augmented Generation represents a significant advancement in
the field of artificial intelligence, merging the best of retrieval-based and
generative technologies to create systems that not only understand and generate
natural language but also deeply comprehend and utilize the vast amounts of
information available in textual form. A RAG vector store is a database or
dataset that contains vectorized data points. 
```
As we can see the output is a huge chunk of text but content of text is not what we need, it tells lots of info but the relevant info with our query seems to be nothing. That is the point we need to enhance for next step.
