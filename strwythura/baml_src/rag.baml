// --- Data Models ---

class Response {
  question_type "verbatim" | "reworded" | "new"
  question string
  answer string | null @description("A concise answer to the question in 1-3 sentences")
}

client<llm> GenericClient {
  provider "openai-generic"
  options {
    model gemma3:12b
    base_url "http://localhost:11434/v1"
    temperature 0.0
  }
}


// --- Functions ---

function RAG(question: string, context: string) -> Response {
  client GenericClient
  prompt #"
    Answer the question using the provided context.
    Do not make up an answer. If you don't know the answer, return a null response.
    Always strive to provide concise, helpful, and context-aware answers.
     
    CONTEXT: {{ context }}

    {{ ctx.output_format }}

    QUESTION: {{ question }}

    RESPONSE:
  "#
}


// --- Tests ---

test TestMissingContext{
  functions [RAG]
  args {
    question "Who founded Lantana Trade LLP?"
    context #"
      BoundaryML is the company that makes BAML, the best way to get structured with LLMs.
    "#
  }
}
