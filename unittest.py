# -*- coding: utf-8 -*-
"""
단위 테스트 
unittest 사용 

Created on Sat Mar  6 17:03:25 2021
@author: jooji
"""
import unittest
import numpy as np
from dezero.core_simple import Variable
import dezero.function as fn


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = fn.square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected) # 두 개체가 동일한지 여부를 판정 이외에도(assertGreater, self.assertTrue등)
        
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = fn.square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)           
        
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = fn.square(x)
        y.backward()
        num_grad = fn.numerical_diff(fn.square, x) # 수치미분 
        flg = np.allclose(x.grad, num_grad) 
        #np.allclose(a,b) : 기본값 |a-b|<=(atol + rtol * |b|) 만족하면 True (값이 얼마나 가까운지 판단)
        self.assertTrue(flg)
        
# $ python -m unittest TEST.py (위에만 했을경우 )

unittest.main()

# $ python TEST.py (main을 호출했을 경우)

# $ python -m unittest discover tests (한꺼번에 테스트할 경우)            