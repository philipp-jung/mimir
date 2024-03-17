import unittest
import pandas as pd
import pdep
import helpers


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
        cls.fds = [pdep.FDTuple(lhs=(lhs,), rhs=rhs) for lhs in range(4) for rhs in range(4) if lhs != rhs]
        cls.fds.append(pdep.FDTuple(lhs=(1, 2), rhs=3))  # (name, zip) -> city
        cls.counts_dict, cls.lhs_values = pdep.fast_fd_counts(cls.df, {}, cls.fds)
        cls.n_rows = cls.df.shape[0]

        # make references to columns more lisible
        cls.zip_code = tuple([2])
        cls.city = 3

    def test_pdep_a(self):
        """
        Test that pdep(city) is computed correctly. We leverage the FD id -> city as context
        for the calculation.
        """
        pdep_city = pdep.pdep_0(self.n_rows, self.counts_dict, 3, (0,))
        if pdep_city is None:
            self.assertEqual(0, 1)
        else:
            self.assertEqual(round(pdep_city, 2), 0.43)

    def test_pdep_a_b(self):
        """
        Test that pdep(zip,city) is computed correctly.
        """
        pdep_zip_city = pdep.pdep(self.n_rows, self.counts_dict, self.lhs_values, self.zip_code, self.city)
        if pdep_zip_city is None:
            self.assertEqual(0, 1)
        else:
            self.assertEqual(round(pdep_zip_city, 2), 0.81)
    
    def test_pdep_a_b_second_order(self):
        """
        Test that pdep((name, zip),city) is computed correctly.
        """
        pdep_name_zip_city = pdep.pdep(self.n_rows, self.counts_dict, self.lhs_values, (1, 2), 3)
        if pdep_name_zip_city is None:
            self.assertEqual(pdep_name_zip_city, None)
        else:
            self.assertEqual(round(pdep_name_zip_city, 2), 1)


    def test_expected_pdep_a_b(self):
        """
        Test that E[pdep(zip,city)] is computed correctly.
        """
        epdep_zip_city = pdep.expected_pdep(self.n_rows, self.counts_dict, self.zip_code, self.city)
        if epdep_zip_city is None:
            self.assertEqual(0, 1)
        else:
            self.assertEqual(round(epdep_zip_city, 2), 0.62)


class TestErrorPdep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame(
            {"id": [1, 2, 3, 4], "name": ["Natalie", "Alice", "Tim", "Bob"]}
        )
        cls.n_rows = cls.df.shape[0]
        cls.fds = [pdep.FDTuple(lhs=(0,), rhs=1), pdep.FDTuple(lhs=(1,), rhs=0)]

    def test_pdep_a_error(self):
        """
        Test that pdep(id) is computed correctly if one of the LHS values is
        marked as an error.
        """
        detected_cells = {(0, 0): '0'}

        error_positions = helpers.ErrorPositions(detected_cells, self.df.shape, {})
        row_errors = error_positions.updated_row_errors()
        
        counts_dict, lhs_values = pdep.fast_fd_counts(self.df, row_errors, self.fds)
        lhs, rhs = self.fds[1].lhs, self.fds[1].rhs
        N = pdep.error_corrected_row_count(self.n_rows, row_errors, lhs, rhs)
        pdep_id = pdep.pdep_0(N, counts_dict, rhs, lhs)
        
        if pdep_id is None:
            self.assertEqual(0, 1)
        else:
            self.assertEqual(round(pdep_id, 2), 0.33)

    def test_pdep_a_all_error(self):
        """
        Test that pdep(id) is computed correctly if one of the LHS values is
        marked as an error.
        """
        detected_cells = {(0, 0): "0", (1, 0): "1", (2, 0): "2", (3, 0): "3"}

        error_positions = helpers.ErrorPositions(detected_cells, self.df.shape, {})
        row_errors = error_positions.updated_row_errors()

        counts_dict, lhs_values = pdep.fast_fd_counts(self.df, row_errors, self.fds)
        lhs, rhs = self.fds[1].lhs, self.fds[1].rhs
        N = pdep.error_corrected_row_count(self.n_rows, row_errors, lhs, rhs)
        pdep_id = pdep.pdep_0(N, counts_dict, rhs, lhs)
        self.assertIsNone(pdep_id)

    def test_pdep_a_b_error_lhs(self):
        """
        Test that pdep(id, name) is computed correctly if two of the LHS values
        are marked as errors.
        """
        detected_cells = {(0, 0): "0", (1, 0): "1"}

        error_positions = helpers.ErrorPositions(detected_cells, self.df.shape, {})
        row_errors = error_positions.updated_row_errors()

        counts_dict, lhs_values = pdep.fast_fd_counts(self.df, row_errors, self.fds)
        lhs, rhs = self.fds[0].lhs, self.fds[0].rhs

        N = pdep.error_corrected_row_count(self.n_rows, row_errors, lhs, rhs)

        pdep_id_name = pdep.pdep(N, counts_dict, lhs_values, lhs, rhs)
        self.assertEqual(1, pdep_id_name)

    def test_pdep_a_b_error_rhs(self):
        """
        Test that pdep(id, name) is computed correctly if two of the RHS values
        are marked as errors.
        """
        detected_cells = {(0, 1): "Otto", (1, 1): "Hanna"}

        error_positions = helpers.ErrorPositions(detected_cells, self.df.shape, {})
        row_errors = error_positions.updated_row_errors()

        counts_dict, lhs_values = pdep.fast_fd_counts(self.df, row_errors, self.fds)
        lhs, rhs = self.fds[0].lhs, self.fds[0].rhs

        N = pdep.error_corrected_row_count(self.n_rows, row_errors, lhs, rhs)
        pdep_id_name = pdep.pdep(N, counts_dict, lhs_values, lhs, rhs)

        self.assertEqual(pdep_id_name, 1)

    def test_pdep_a_b_all_error_lhs(self):
        """
        Test that pdep(id, name) is None if all values in A are errors.
        """
        detected_cells = {(0, 0): "0", (1, 0): "1", (2, 0): "2", (3, 0): "3"}
        
        error_positions = helpers.ErrorPositions(detected_cells, self.df.shape, {})
        row_errors = error_positions.updated_row_errors()

        counts_dict, lhs_values = pdep.fast_fd_counts(self.df, row_errors, self.fds)
        lhs, rhs = self.fds[0].lhs, self.fds[0].rhs

        N = pdep.error_corrected_row_count(self.n_rows, row_errors, lhs, rhs)
        pdep_id_name = pdep.pdep(N, counts_dict, lhs_values, lhs, rhs)

        self.assertIsNone(pdep_id_name)

    def test_pdep_a_b_all_error_rhs(self):
        """
        Test that pdep(id, name) is None if all values in A are errors.
        """
        detected_cells = {(0, 0): "Otto", (1, 0): "Hanna", (2, 0): "Peter", (3, 0): "Ella"}

        error_positions = helpers.ErrorPositions(detected_cells, self.df.shape, {})
        row_errors = error_positions.updated_row_errors()

        counts_dict, lhs_values = pdep.fast_fd_counts(self.df, row_errors, self.fds)
        lhs, rhs = self.fds[0].lhs, self.fds[0].rhs

        N = pdep.error_corrected_row_count(self.n_rows, row_errors, lhs, rhs)
        pdep_id_name = pdep.pdep(N, counts_dict, lhs_values, lhs, rhs)

        self.assertIsNone(pdep_id_name)


if __name__ == "__main__":
    unittest.main()
