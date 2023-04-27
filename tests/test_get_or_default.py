import unittest

from mgt.models.utils import get_or_default


class TestGetOrDefault(unittest.TestCase):

    def test_key_exists_in_dictionary(self):
        dictionary = {"key1": "value1", "key2": "value2"}
        defaults = {"key1": "default_value1", "key2": "default_value2", "key3": "default_value3"}
        result = get_or_default(dictionary, "key1", defaults)
        self.assertEqual(result, "value1", "Value should be retrieved from the dictionary when the key exists")

    def test_key_not_exists_in_dictionary(self):
        dictionary = {"key1": "value1", "key2": "value2"}
        defaults = {"key1": "default_value1", "key2": "default_value2", "key3": "default_value3"}
        result = get_or_default(dictionary, "key3", defaults)
        self.assertEqual(result, "default_value3",
                         "Default value should be returned when the key does not exist in the dictionary")

    def test_key_exists_in_both_dictionaries(self):
        dictionary = {"key1": "value1", "key2": "value2"}
        defaults = {"key1": "default_value1", "key2": "default_value2"}
        result = get_or_default(dictionary, "key2", defaults)
        self.assertEqual(result, "value2",
                         "Value should be retrieved from the dictionary when the key exists in both the dictionary and defaults")

    def test_key_not_exists_in_either_dictionary(self):
        dictionary = {"key1": "value1", "key2": "value2"}
        defaults = {"key1": "default_value1", "key3": "default_value3"}
        with self.assertRaises(KeyError):
            get_or_default(dictionary, "key4", defaults)

    def test_different_types(self):
        dictionary = {1: "value1", 2: "value2", "key1": 100, "key2": 200}
        defaults = {1: "default_value1", 2: "default_value2", "key1": 300, "key2": 400, "key3": "default_value3"}

        result_int_key = get_or_default(dictionary, 1, defaults)
        self.assertEqual(result_int_key, "value1", "Value should be retrieved from the dictionary for an integer key")

        result_str_key = get_or_default(dictionary, "key1", defaults)
        self.assertEqual(result_str_key, 100, "Value should be retrieved from the dictionary for a string key")

        result_default_str_key = get_or_default(dictionary, "key3", defaults)
        self.assertEqual(result_default_str_key, "default_value3", "Default value should be returned for a string key not in the dictionary")

        with self.assertRaises(KeyError):
            get_or_default(dictionary, 3, defaults)

if __name__ == '__main__':
    unittest.main()
