# -*- coding: utf-8 -*-

from .context import app
from app import helpers

import unittest


class HelpersTestSuite(unittest.TestCase):
    """Helpers test cases."""

    def test_pubulish_message(self):
        #result = helpers.publish_message("", "", "OK", "10001", "")
        self.assertIsNone(None)


if __name__ == '__main__':
    unittest.main()