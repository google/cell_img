# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for cell_img. Verifies cell_img can be imported.

"""

import unittest
import cell_img
from cell_img import data_utils


class CellImgTest(unittest.TestCase):

  def test_import(self):
    self.assertTrue(cell_img)
    self.assertTrue(data_utils)

if __name__ == '__main__':
  unittest.main()
