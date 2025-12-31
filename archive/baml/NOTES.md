import baml_py
from strwythura.baml_client import types as baml_types

response: baml_types.Response = rag.qa_cycle(

        except baml_py.internal_monkeypatch.BamlValidationError as baml_ex:
            ic(ex)



<details>
  <summary>Developer Notes</summary>

After each `BAML` release update, some committer needs to regenerate
its Python client source:

```bash
poetry run baml-cli generate --from strwythura/baml_src
```
</details>



from .baml_client import b
from .baml_client import types as baml_types

    os.environ["BAML_LOG"] = "WARN"


    ) -> baml_types.Response:

    response: baml_types.Response = b.RAG(
        question,
        context,
    )
