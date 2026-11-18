"""Verify the metadata exposed through the Hermes provider interface."""
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from support import NAME, ROOT


class ProviderMetadataTests(unittest.TestCase):
    def test_context_window_matches_provider_lookup(self):
        class ProviderProfile:
            def __init__(self, **kwargs):
                self.model_capabilities = kwargs["model_capabilities"]

        modules = {
            "providers": SimpleNamespace(register_provider=lambda profile: None),
            "providers.base": SimpleNamespace(ProviderProfile=ProviderProfile),
        }
        with patch.dict(sys.modules, modules):
            profile = runpy.run_path(str(ROOT / "__init__.py"), run_name=NAME + ".profile")["profile"]

        for model, expected in (
            ("opencode/deepseek-v4.1-flash-free", 200_000),
            ("opencode/glm-5.3-flash-free", 131_072),
        ):
            with self.subTest(model=model):
                self.assertEqual(profile.get_model_context_length(model), expected)
                self.assertEqual(profile.model_capabilities[model]["context_window"], expected)

        self.assertIsNone(profile.get_model_context_length("uncurated/model"))
        self.assertNotIn("uncurated/model", profile.model_capabilities)
