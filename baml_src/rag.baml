class Response {
  question string
  answer string
}

client<llm> Llama3Client {
  provider "openai-generic"
  options {
    base_url "http://localhost:11434/v1"
    model llama3
  }
}

function RAG(question: string, context: string) -> Response {
  client Llama3Client
  prompt #"
    Answer the question in full sentences using the provided context.
    Do not make up an answer. If the information is not provided in the context, say so clearly.
    
    QUESTION: {{ question }}
    RELEVANT CONTEXT: {{ context }}

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
