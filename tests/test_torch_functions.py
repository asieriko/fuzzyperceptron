import unittest
import torch

from src import torch_functions as torch_functions

# TODO: Falta el caso de x_n_m */-+ w_1_1
'''
Formato de salida
un caso de entrada y una neurona de sailda
[
    [a]
]
un caso de entrada y dos neuronas de sailda
[
    [a, b]
]
dos casos de entrada y una neurona de salida

[
    [a],
    [c]
]
dos casos de entrada y tres neuronas de salida

[
    [a, b, c],
    [c, d, e]
]
'''
'''
x_1_1 | 1,5 | 2
x_1_m | 2,1,5 | 3
x_n_1 | 3,5 | 2
x_n_m | 2,3,5 | 3
'''


class TestTorchUnaryFunctions(unittest.TestCase):
    """
    producto de input por pesos
    salida esperada: lista con una lista por neurona y dentro una por caso de input
    una neurona 1 caso: [[[caso]]]
    dos neuronas 1 caso: [[[caso]],[[caso2]]]
    una neurona x casos:[[[caso],[caso]]]
    dos neurona x casos:[[[caso],[caso]],[[caso],[caso]]]
    """

    x_1_1 = torch.Tensor(  # Una neurona, un caso
        [
            [1, 2, 3, 4, 5]
        ]
    )

    x_1_m = torch.Tensor(  # dos neurona, un caso -- No hace falta, el de abajo es igual
        [
            [
                [1, 2, 3, 4, 5]
            ],
            [
                [6, 7, 8, 9, 10]
            ]
        ]
    )

    x_n_1 = torch.Tensor(  # Una neurona, tres casos
        [
            [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5]
            ]
        ]
    )

    x_n_m = torch.Tensor(  # Dos neurona, tres casos
        [
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]
            ],
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]
            ]
        ]
    )

    def test_vmean_1_1(self):
        expected = torch.Tensor(
            [
                [3]
            ]
        )
        result = torch_functions.vmean(self.x_1_1)
        self.assertEqual(result, expected)

    def test_vmean_1_m(self):
        expected = torch.Tensor(
            [
                [3, 8]
            ]
        )
        result = torch_functions.vmean(self.x_1_m)
        torch.testing.assert_allclose(result, expected)

    def test_vmean_n_1(self):
        expected = torch.Tensor(
            [
                [3],
                [3],
                [3]
            ]
        )
        result = torch_functions.vmean(self.x_n_1)
        torch.testing.assert_allclose(result, expected)

    def test_vmean_n_m(self):
        expected = torch.Tensor(
            [
                [1, 1],
                [1, 1],
                [1, 1]
            ]
        )
        result = torch_functions.vmean(self.x_n_m)
        torch.testing.assert_allclose(result, expected)

    def test_vsum_1_1(self):
        expected = torch.Tensor(
            [
                [15]
            ]
        )
        result = torch_functions.vsum(self.x_1_1)
        self.assertEqual(result, expected)

    def test_vsum_1_m(self):
        expected = torch.Tensor(
            [
                [15, 40]
            ]
        )
        result = torch_functions.vsum(self.x_1_m)
        torch.testing.assert_allclose(result, expected)

    def test_vsum_n_1(self):
        expected = torch.Tensor(
            [
                [15],
                [15],
                [15]
            ]
        )
        result = torch_functions.vsum(self.x_n_1)
        torch.testing.assert_allclose(result, expected)

    def test_vsum_n_m(self):
        expected = torch.Tensor(
            [
                [5, 5],
                [5, 5],
                [5, 5]
            ]
        )
        result = torch_functions.vsum(self.x_n_m)
        torch.testing.assert_allclose(result, expected)

    def test_max_1_1(self):
        expected = torch.Tensor(
            [
                [5]
            ]
        )
        result = torch_functions.vmax(self.x_1_1)
        self.assertEqual(result, expected)

    def test_vmax_1_m(self):
        expected = torch.Tensor(
            [
                [5, 10]
            ]
        )
        result = torch_functions.vmax(self.x_1_m)
        torch.testing.assert_allclose(result, expected)

    def test_vmax_n_1(self):
        expected = torch.Tensor(
            [
                [5],
                [5],
                [5]
            ]
        )
        result = torch_functions.vmax(self.x_n_1)
        torch.testing.assert_allclose(result, expected)

    def test_vmax_n_m(self):
        expected = torch.Tensor(
            [
                [1, 1],
                [1, 1],
                [1, 1]
            ]
        )
        result = torch_functions.vmax(self.x_n_m)
        torch.testing.assert_allclose(result, expected)

    def test_min_1_1(self):
        expected = torch.Tensor(
            [
                [1]
            ]
        )
        result = torch_functions.vmin(self.x_1_1)
        self.assertEqual(result, expected)

    def test_vmin_1_m(self):
        expected = torch.Tensor(
            [
                [1, 6]
            ]
        )
        result = torch_functions.vmin(self.x_1_m)
        torch.testing.assert_allclose(result, expected)

    def test_vmin_n_1(self):
        expected = torch.Tensor(
            [
                [1],
                [1],
                [1]
            ]
        )
        result = torch_functions.vmin(self.x_n_1)
        torch.testing.assert_allclose(result, expected)

    def test_vmin_n_m(self):
        expected = torch.Tensor(
            [
                [1, 1],
                [1, 1],
                [1, 1]
            ]
        )
        result = torch_functions.vmin(self.x_n_m)
        torch.testing.assert_allclose(result, expected)


class TestTorchBinaryFunctions(unittest.TestCase):
    x_1_1 = torch.Tensor(  # Una neurona, un caso
        [
            [1, 2, 3, 4, 5]
        ]
    )

    x_1_m = torch.Tensor(  # dos neurona, un caso para B
        [
            [
                [1, 2, 3, 4, 5]
            ],
            [
                [6, 7, 8, 9, 10]
            ]
        ]
    )

    x_n_1 = torch.Tensor(  # Una neurona, tres casos
        [
            [  # para B
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5]
            ]
        ]
    )

    x_n_m = torch.Tensor(  # Dos neurona, tres caso  para B
        [
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]
            ],
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]
            ]
        ]
    )

    w_1 = torch.Tensor(  # Pesos para una neurona de salida
        [
            [1, 1, 1, 1, 1]
        ]
    )

    w_n = torch.Tensor(  # Pesos para dos neuronas de salida
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ]
    )

    def test_dot_1_1(self):
        expected = torch.Tensor(
            [
                [
                    [1, 2, 3, 4, 5]
                ]
            ]
        )
        result = torch_functions.dot(self.x_1_1, self.w_1)
        torch.testing.assert_allclose(result, expected)

    def test_dot_1_n(self):
        expected = torch.Tensor(
            [
                [
                    [1, 2, 3, 4, 5]
                ],
                [
                    [1, 2, 3, 4, 5]
                ]
            ]
        )
        result = torch_functions.dot(self.x_1_1, self.w_n)
        torch.testing.assert_allclose(result, expected)

    def test_dot_n_1(self):
        expected = torch.Tensor(
            [
                [
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5]
                ]
            ]
        )
        result = torch_functions.dot(self.x_n_1, self.w_1)
        torch.testing.assert_allclose(result, expected)

    def test_dot_1_m(self):
        expected = torch.Tensor(
            [
                [
                    [1, 2, 3, 4, 5]
                ],
                [
                    [6, 7, 8, 9, 10]
                ]
            ]
        )
        result = torch_functions.dot(self.x_1_m, self.w_n)
        torch.testing.assert_allclose(result, expected)

    def test_dot_n_m(self):
        expected = torch.Tensor(
            [
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]
                ],
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]
                ]
            ]
        )
        result = torch_functions.dot(self.x_n_m, self.w_n)
        torch.testing.assert_allclose(result, expected)

    def test_div_1_1(self):
        expected = torch.Tensor(
            [
                [
                    [1, 2, 3, 4, 5]
                ]
            ]
        )
        result = torch_functions.div(self.x_1_1, self.w_1)
        torch.testing.assert_allclose(result, expected)

    def test_div_1_n(self):
        expected = torch.Tensor(
            [
                [
                    [1, 2, 3, 4, 5]
                ],
                [
                    [1, 2, 3, 4, 5]
                ]
            ]
        )
        result = torch_functions.div(self.x_1_1, self.w_n)
        torch.testing.assert_allclose(result, expected)

    def test_div_n_1(self):
        expected = torch.Tensor(
            [
                [
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5]
                ]
            ]
        )
        result = torch_functions.div(self.x_n_1, self.w_1)
        torch.testing.assert_allclose(result, expected)

    def test_div_1_m(self):
        expected = torch.Tensor(
            [
                [
                    [1, 2, 3, 4, 5]
                ],
                [
                    [6, 7, 8, 9, 10]
                ]
            ]
        )
        result = torch_functions.div(self.x_1_m, self.w_n)
        torch.testing.assert_allclose(result, expected)

    def test_div_n_m(self):
        expected = torch.Tensor(
            [
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]
                ],
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]
                ]
            ]
        )
        result = torch_functions.div(self.x_n_m, self.w_n)
        torch.testing.assert_allclose(result, expected)

    def test_madd_1_1(self):
        expected = torch.Tensor(
            [
                [
                    [2, 3, 4, 5, 6]
                ]
            ]
        )
        result = torch_functions.madd(self.x_1_1, self.w_1)
        torch.testing.assert_allclose(result, expected)

    def test_madd_1_n(self):
        expected = torch.Tensor(
            [
                [
                    [2, 3, 4, 5, 6]
                ],
                [
                    [2, 3, 4, 5, 6]
                ]
            ]
        )
        result = torch_functions.madd(self.x_1_1, self.w_n)
        torch.testing.assert_allclose(result, expected)

    def test_madd_n_1(self):
        expected = torch.Tensor(
            [
                [
                    [2, 3, 4, 5, 6],
                    [2, 3, 4, 5, 6],
                    [2, 3, 4, 5, 6]
                ]
            ]
        )
        result = torch_functions.madd(self.x_n_1, self.w_1)
        torch.testing.assert_allclose(result, expected)

    def test_madd_1_m(self):
        expected = torch.Tensor(
            [
                [
                    [2, 3, 4, 5, 6]
                ],
                [
                    [2, 3, 4, 5, 6]
                ]
            ]
        )
        result = torch_functions.madd(self.x_1_1, self.w_n)
        torch.testing.assert_allclose(result, expected)

    def test_madd_n_m(self):
        expected = torch.Tensor(
            [
                [
                    [2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2]
                ],
                [
                    [2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2]
                ]
            ]
        )
        result = torch_functions.madd(self.x_n_m, self.w_n)
        torch.testing.assert_allclose(result, expected)

    def test_mminus_1_1(self):
        expected = torch.Tensor(
            [
                [
                    [0, 1, 2, 3, 4]
                ]
            ]
        )
        result = torch_functions.mminus(self.x_1_1, self.w_1)
        torch.testing.assert_allclose(result, expected)

    def test_mminus_1_n(self):
        expected = torch.Tensor(
            [
                [
                    [0, 1, 2, 3, 4]
                ],
                [
                    [0, 1, 2, 3, 4]
                ]
            ]
        )
        result = torch_functions.mminus(self.x_1_1, self.w_n)
        torch.testing.assert_allclose(result, expected)

    def test_mminus_n_1(self):
        expected = torch.Tensor(
            [
                [
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4]
                ]
            ]
        )
        result = torch_functions.mminus(self.x_n_1, self.w_1)
        torch.testing.assert_allclose(result, expected)

    def test_mminus_1_m(self):
        expected = torch.Tensor(
            [
                [
                    [0, 1, 2, 3, 4]
                ],
                [
                    [0, 1, 2, 3, 4]
                ]
            ]
        )
        result = torch_functions.mminus(self.x_1_1, self.w_n)
        torch.testing.assert_allclose(result, expected)

    def test_mminus_n_m(self):
        expected = torch.Tensor(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]
                ]
            ]
        )
        result = torch_functions.mminus(self.x_n_m, self.w_n)
        torch.testing.assert_allclose(result, expected)


class TestTorchComposedFunctions(unittest.TestCase):
    x_1_1 = torch.Tensor(
        [
            [1, 2, 3, 4, 5]
        ]
    )

    x_1_m = torch.Tensor(
        [
            [
                [1, 1, 1, 1, 1]
            ],
            [
                [1, 1, 1, 1, 1]
            ]
        ]
    )

    x_n_1 = torch.Tensor(
        [
            [
                [1, 1, 1, 1, 1]
            ],
            [
                [1, 1, 1, 1, 1]
            ],
            [
                [1, 1, 1, 1, 1]
            ]
        ]
    )

    x_n_m = torch.Tensor(
        [
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]
            ],
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]
            ]
        ]
    )

    x1_n_1 = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5]
        ]
    )

    w_1 = torch.Tensor(
        [
            [1, 1, 1, 1, 1]
        ]
    )

    w_n = torch.Tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ]
    )

    def test_dotsum_1_1(self):
        expected = torch.Tensor(
            [
                [15]
            ]
        )
        result = torch_functions.vsum(torch_functions.dot(self.x_1_1, self.w_1))
        self.assertEqual(result, expected)

    def test_dotsum_1_m(self):
        expected = torch.Tensor(
            [
                [5, 5]
            ]
        )
        result = torch_functions.vsum(torch_functions.dot(self.x_1_m, self.w_n))
        torch.testing.assert_allclose(result, expected)

    def test_dotsum_n_1(self):
        expected = torch.Tensor(
            [
                [15],
                [15],
                [15]
            ]
        )
        result = torch_functions.vsum(torch_functions.dot(self.x1_n_1, self.w_1))
        torch.testing.assert_allclose(result, expected)
        # self.assertEqual(result,expected)

    def test_dotsum_n_m(self):
        expected = torch.Tensor(
            [
                [5, 5],
                [5, 5],
                [5, 5]
            ]
        )
        result = torch_functions.vsum(torch_functions.dot(self.x_n_m, self.w_n))
        torch.testing.assert_allclose(result, expected)

    def test_dotsum_linear_1_1(self):
        m = torch.nn.Linear(5, 1, False)
        m.weight = torch.nn.Parameter(self.w_1)
        expected = m(self.x_1_1)

        result = torch_functions.vsum(torch_functions.dot(self.x_1_1, self.w_1))
        self.assertEqual(result, expected)

    def test_dotsum_linear_1_m(self):
        m = torch.nn.Linear(5, 2, False)
        m.weight = torch.nn.Parameter(self.w_n)
        expected = m(self.x_1_1)

        result = torch_functions.vsum(torch_functions.dot(self.x_1_1, self.w_n))
        torch.testing.assert_allclose(result, expected)

    def test_dotsum_linear_n_1(self):
        m = torch.nn.Linear(5, 1, False)
        m.weight = torch.nn.Parameter(self.w_1)
        expected = m(self.x1_n_1)

        result = torch_functions.vsum(torch_functions.dot(self.x1_n_1, self.w_1))
        torch.testing.assert_allclose(result, expected)
        # self.assertEqual(result,expected)

    def test_dotsum_linear_n_m(self):
        m = torch.nn.Linear(5, 2, False)
        m.weight = torch.nn.Parameter(self.w_n)
        expected = m(self.x1_n_1)

        result = torch_functions.vsum(torch_functions.dot(self.x1_n_1, self.w_n))
        torch.testing.assert_allclose(result, expected)


if __name__ == '__main__':
    unittest.main()
