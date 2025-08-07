class Response {
  question_type "verbatim" | "reworded" | "new"
  question string
  answer string
}

client<llm> Llama3Client {
  provider "openai-generic"
  options {
    base_url "http://localhost:11434/v1"
    model gemma3:12b
  }
}

function RAG(question: string, context: string) -> Response {
  client Llama3Client
  prompt #"
    Answer the question in full sentences using the provided context.
    Do not make up an answer. If the information is not provided in the context, say so clearly.
     
    RELEVANT CONTEXT: {{ context }}
    QUESTION: {{ question }}

    {{ ctx.output_format }}

    RESPONSE:
  "#
}

test TestMissingContext{
  functions [RAG]
  args {
    question "Who founded Lantana Trade LLP?"
    context #"
      BoundaryML is the company that makes BAML, the best way to get structured with LLMs.
    "#
  }
}
