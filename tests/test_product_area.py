from aegis.ingest.product_area import derive_product_area_from_doc_path, derive_product_area_from_text


def test_security_subdirectory():
    assert derive_product_area_from_doc_path("tutorial/security/first-steps.md") == "security"
    assert derive_product_area_from_doc_path("advanced/security/oauth2-scopes.md") == "security"


def test_dependencies():
    assert derive_product_area_from_doc_path("tutorial/dependencies/index.md") == "dependencies"
    assert derive_product_area_from_doc_path("advanced/advanced-dependencies.md") == "dependencies"


def test_deployment_directory_catches_everything_inside():
    assert derive_product_area_from_doc_path("deployment/docker.md") == "deployment"
    assert derive_product_area_from_doc_path("deployment/https.md") == "deployment"


def test_request_data_bucket():
    assert derive_product_area_from_doc_path("tutorial/body.md") == "request-data"
    assert derive_product_area_from_doc_path("tutorial/cookie-params.md") == "request-data"


def test_fallback_to_top_level_directory():
    assert derive_product_area_from_doc_path("tutorial/first-steps.md") == "core"
    assert derive_product_area_from_doc_path("advanced/dataclasses.md") == "advanced"


def test_fallback_to_general_for_root_files():
    assert derive_product_area_from_doc_path("python-types.md") == "general"


def test_generic_english_word_response_does_not_false_positive():
    # "response" alone is too generic ("no response from server", "in
    # response to your comment") -- measured false positives on real issues.
    assert derive_product_area_from_text("Connection slows over time", "Getting no response after 30s") == "general"


def test_specific_response_terms_still_match():
    assert derive_product_area_from_text("Bug", "response_model validation fails silently") == "responses"


def test_text_matches_known_keyword():
    assert derive_product_area_from_text("OAuth2 login fails with 401", "using Security() dependency") == "security"


def test_text_falls_back_to_general_when_no_match():
    assert derive_product_area_from_text("Typo in a random error message", "") == "general"


def test_text_ignores_keywords_inside_code_blocks():
    body = "Schema generation is broken.\n\n```python\nasync def read_items():\n    ...\n```"
    assert derive_product_area_from_text("OpenAPI schema bug", body) == "openapi"


def test_text_ignores_issue_template_checklist_boilerplate():
    # FastAPI's old issue template checklist mentions Pydantic/Swagger UI/ReDoc
    # by name on ~40% of all closed issues regardless of topic (measured). No
    # other keyword appears here, so without stripping, this would falsely
    # classify as "openapi" (from "Swagger UI") purely from template text.
    body = (
        "### First Check\n\n"
        "- [X] I already checked if it is not related to FastAPI but to [Pydantic](https://x).\n"
        "- [X] I already checked if it is not related to FastAPI but to [Swagger UI](https://x).\n"
        "- [X] I already checked if it is not related to FastAPI but to [ReDoc](https://x).\n\n"
        "### Description\n\nMy cron job scheduler feature idea for the framework."
    )
    assert derive_product_area_from_text("Feature idea", body) == "general"
