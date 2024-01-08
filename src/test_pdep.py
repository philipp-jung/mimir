import unittest
import pandas as pd
import pdep


class TestPdep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6, 7],
                "name": ["Natalie", "Alice", "Tim", "Bob", "Bob", "Alice", "Bob"],
                "zip": [14193, 14193, 14880, 14882, 14882, 14880, 14193],
                "city": [
                    "Berlin",
                    "Berlin",
                    "Potsdam",
                    "Potsdam",
                    "Potsdam",
                    "Potsdam",
                    "Berln",
                ],
            }
        )
        cls.counts_dict = {1: pdep.mine_all_counts(cls.df, [{}], order=1)}
        cls.n_rows = cls.df.shape[0]

        # make references to columns more lisible
        cls.zip_code = tuple([2])
        cls.city = 3

    def test_pdep_a(self):
        """
        Test that pdep(city) is computed correctly.
        """
        pdep_city = round(
            pdep.pdep(self.n_rows, self.counts_dict, {}, 1, tuple([self.city])), 2
        )
        self.assertEqual(pdep_city, 0.43)

    def test_pdep_a_b(self):
        """
        Test that pdep(zip,city) is computed correctly.
        """
        pdep_zip_city = round(
            pdep.pdep(self.n_rows, self.counts_dict, {}, 1, self.zip_code, self.city), 2
        )
        self.assertEqual(pdep_zip_city, 0.81)

    def test_expected_pdep_a_b(self):
        """
        Test that E[pdep(zip,city)] is computed correctly.
        """
        epdep_zip_city = round(
            pdep.expected_pdep(
                self.n_rows, self.counts_dict, {}, 1, self.zip_code, self.city
            ),
            2,
        )
        self.assertEqual(epdep_zip_city, 0.62)


class TestErrorPdep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame(
            {"id": [1, 2, 3, 4], "name": ["Natalie", "Alice", "Tim", "Bob"]}
        )
        cls.n_rows = cls.df.shape[0]

    def test_pdep_a_error(self):
        """
        Test that pdep(id) is computed correctly if one of the LHS values is
        marked as an error.
        """
        detected_cells = {(0, 0): "0"}
        counts_dict = {1: pdep.mine_all_counts(self.df, detected_cells, order=1)}
        pdep_id = round(pdep.pdep(self.n_rows, counts_dict, detected_cells, 1, tuple([0])), 2)
        self.assertEqual(pdep_id, 0.33)

    def test_pdep_a_all_error(self):
        """
        Test that pdep(id) is computed correctly if one of the LHS values is
        marked as an error.
        """
        detected_cells = {(0, 0): "0", (1, 0): "1", (2, 0): "2", (3, 0): "3"}
        counts_dict = {1: pdep.mine_all_counts(self.df, detected_cells, order=1)}
        pdep_id = pdep.pdep(self.n_rows, counts_dict, detected_cells, 1, tuple([0]))
        self.assertIsNone(pdep_id)

    def test_pdep_a_b_error_lhs(self):
        """
        Test that pdep(id, name) is computed correctly if two of the LHS values
        are marked as errors.
        """
        detected_cells = {(0, 0): "0", (1, 0): "1"}
        counts_dict = {1: pdep.mine_all_counts(self.df, detected_cells, order=1)}
        pdep_a_b = round(pdep.pdep(self.n_rows, counts_dict, detected_cells, 1, tuple([0]), 1), 2)
        self.assertEqual(1, pdep_a_b)

    def test_pdep_a_b_error_rhs(self):
        """
        Test that pdep(id, name) is computed correctly if two of the RHS values
        are marked as errors.
        """
        detected_cells = {(0, 1): "Otto", (1, 1): "Hanna"}
        counts_dict = {1: pdep.mine_all_counts(self.df, detected_cells, order=1)}
        pdep_a_b = round(pdep.pdep(self.n_rows, counts_dict, detected_cells, 1, tuple([0]), 1), 2)
        self.assertEqual(pdep_a_b, 1)

    def test_pdep_a_b_all_error_lhs(self):
        """
        Test that pdep(id, name) is None if all values in A are errors.
        """
        detected_cells = {(0, 0): "0", (1, 0): "1", (2, 0): "2", (3, 0): "3"}
        counts_dict = {1: pdep.mine_all_counts(self.df, detected_cells, order=1)}
        pdep_a_b = pdep.pdep(self.n_rows, counts_dict, detected_cells, 1, tuple([0]), 1)
        self.assertIsNone(pdep_a_b)

    def test_pdep_a_b_all_error_rhs(self):
        """
        Test that pdep(id, name) is None if all values in A are errors.
        """
        detected_cells = {(0, 0): "Otto", (1, 0): "Hanna", (2, 0): "Peter", (3, 0): "Ella"}
        counts_dict = {1: pdep.mine_all_counts(self.df, detected_cells, order=1)}
        pdep_a_b = pdep.pdep(self.n_rows, counts_dict, detected_cells, 1, tuple([0]), 1)
        self.assertIsNone(pdep_a_b)


if __name__ == "__main__":
    unittest.main()
