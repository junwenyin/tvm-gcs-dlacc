# -*- coding: utf-8 -*-

from .context import app
from app import main

import unittest


class MainTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_main(self):
        self.assertIsNone(None)


if __name__ == '__main__':
    unittest.main()