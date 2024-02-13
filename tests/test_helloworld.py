import unittest

from project.helloworld import hello

class TestHelloWorld(unittest.TestCase):
    def test_hello_world(self):
        self.assertEqual(hello(), "Hello, World!")

    def test_hello_name(self):
        self.assertEqual(hello("Alice"), "Hello, Alice!")
    