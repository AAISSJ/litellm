import pytest

from litellm.llms.cohere.embed.transformation import CohereEmbeddingConfig


class TestCohereEmbeddingV2MapOpenAIParams:
    """The OpenAI SDK sends encoding_format="base64" whenever the caller omits it,
    and Cohere's embedding_types enum has no "base64", so passing it through
    verbatim gets a 400 from Cohere. LIT-6480, direct-provider twin of #38670."""

    @pytest.mark.parametrize(
        "encoding_format,expected_embedding_types",
        [
            ("float", ["float"]),
            ("base64", ["float"]),
            (["float", "int8"], ["float", "int8"]),
            (["base64", "int8"], ["float", "int8"]),
            (["float", "base64"], ["float"]),
        ],
    )
    def test_encoding_format_maps_to_valid_embedding_types(self, encoding_format, expected_embedding_types):
        optional_params = CohereEmbeddingConfig().map_openai_params(
            non_default_params={"encoding_format": encoding_format},
            optional_params={},
            model="embed-english-v3.0",
        )
        assert optional_params["embedding_types"] == expected_embedding_types

    def test_dimensions_maps_to_output_dimension(self):
        optional_params = CohereEmbeddingConfig().map_openai_params(
            non_default_params={"dimensions": 512},
            optional_params={},
            model="embed-v4.0",
        )
        assert optional_params["output_dimension"] == 512
