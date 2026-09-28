"""Basic checks for the installed package."""

import unittest
from importlib.metadata import version
import numcompute

class TestPackage(unittest.TestCase):
    def test_package_version(self):
        self.assertEqual(numcompute.__version__, "0.1.0")

    def test_installed_version_matches(self):
        self.assertEqual(version("numcompute"),numcompute.__version__,)



if __name__ == "__main__":
    unittest.main()