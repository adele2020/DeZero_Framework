# -*- coding: utf-8 -*-
"""
Variable 클래스

Created on Sat Mar  6 11:09:07 2021
@author: joojg
"""
import weakref
import numpy as np
import contextlib

#--------------------------------------------------------------------------------------
# Config
#-------------------------------------------------------------------------------------- 
class Config:
    enable_backprop = True
    
@contextlib.contextmanager #문맥을 판단하는 함수 
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop',False) #역전파가 필요없을때 with블록에서 순전파코드만실행   

# numpy의 다차원배열만 취급(numpy.ndarray)
# 다차원 배열(텐서)의 예 : 스칼라(0차원), 벡터(1차원), 행렬(다차원)
# 넘파이의 ndarray 인스턴스에는 ndim이라는 인스턴스 변수가 있다 
# 인스턴스변수(각인스턴스가자신만의복사본소유)vs멤버변수(모든인스턴스와공유되는변수사본이하) 
# 벡터의 차원 (ex. 3차원 벡터) : 벡터의 원소 수 
# 배열의 차원 (ex. 3차원 배열) : (원소가 아닌)축이 3개
# 역전파를 이용하면 미분을 효율적으로 계산할 수 있고 결과값의 오차도 더 작다.
# 전파되는 데이터는 모두 y의 미분값 즉 'y의 ooo에 대한 미분값'이 전파
# 벡터나 행렬 등 다변수에 대한 미분은 기울기(gradient)라고 함. 
#--------------------------------------------------------------------------------------
# Variable
#-------------------------------------------------------------------------------------- 
class Variable:
    __array_priority = 200
    
    def __init__(self, data, name=None):
        # ndarray 인스턴스만 취급
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
        self.data = data # 인스턴스 변수 
        self.name = name
        self.grad = None 
        self.creator = None
        self.generation = 0 # 세대 수를 기록하는 변수 
    
    # Variable은 ndarray만 취급한다. 다차원배열용 인스턴스들을 만들어준다.
    # @property 한 줄 덕분에 메서드를 인스턴스 변수처럼 사용
    @property 
    def shape(self):
        return self.data.shape # 차원의 수
    
    @property
    def size(self):
        return self.data.size # 원소 수
    
    @property
    def dtype(self):
        return self.data.dtype # 데이터 타입
    
    # 특수메서드를 구현하면 Variable 인스턴스에 대해서도 len함수를 사용할 수 있게 된다.
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n','\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    def set_creator(self, func):
        self.creator = func #함수와변수를연결  
        self.generation = func.generation + 1 # 세대를 기록한다(부모 세대 + 1)
       
    # Define-by-Run : 딥러닝에서 수행하는 계산들을 계산 시점에 '연결'하는 방식으로 '동적 계산 그래프'라고 함.
    # 자동미분의기초
    # 1. 함수를 가져온다.
    # 2. 함수의 입력을 가져온다. 
    # 3. 함수의 backward 메서드를 호출한다. 
    #   x    -> [f]     ->  self
    # input  <- creator <- 
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data) #self.data가 스칼라면 self.grad도 스칼라 
            
        funcs = []
        seen_set = set()
        
        def add_func(f): #중첩함수로 정의 
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x : x.generation)
                
        # 재귀를 사용한 구현
        #f = self.creator() #자신을 생성한 함수
        #if f is not None:
        #    x = f.input #자신을 생성한 함수의 input
        #    x.grad = f.backward(self.grad) #지금 자신의 grad와 자신을 생성한 함수의 backward를 태워 자신보다 하나앞 grad 생성
        #    x.backward() # 자신보다 하나 앞 backward() 호출
        
        # 반복문을 이용한 구현
        #복잡한 계산 그래프를 부드럽게 확장하여 구현할 수 있게 처리 효율이 있다. 
                
        #funcs = [self.creator]
        
        add_func(self.creator) # generation을 추가 
        
        while funcs:
            f = funcs.pop()          
            # 입출력을 하나일 때로 한정함.        
            #x, y = f.input, f.output
            #x.grad = f.backward(y.grad)
            
            #if x.creator is not None:
            #    funcs.append(x.creator)
            #   
            gys = [output().grad for output in f.outputs] # 출력변수인 outputs에 담겨 있는 미분값들을 리스트에 담음 
                                                          # 수정 전 : output.grad 순환참조문제를 해결하기 위해 수정 
            gxs = f.backward(*gys) # 함수 f의 역전파 호출(gys리스트 언팩)
            if not isinstance(gxs, tuple): # 튜플이 아니면 튜플로 반환 
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None: # 미분값을 그냥 덮어쓰게 되는 것을 방지 
                    x.grad = gx 
                else:
                    x.grad = x.grad + gx
                
                if x.creator is not None:
                    add_func(x.creator) # 수정 전 : funcs.append(x.creator)
        
        if not retain_grad: 
            for y in f.outputs:
                y().grad = None # y는 약한 참조(weakref) # 각 함수의 출력 변수의 미분값을 유지하지 않도록 
                    
    def cleargrad(self):# Variable에 담기 x를 재사용하는 코드에서 미분값이 누적되어 잘못 출력되는 것을 방지 
        self.grad = None                    
                
# 계산그래프 : 원과 사각형 모양의 노드들을 화살표로 연결해 계산 과정 표현
# 그래프 : 노드와 엣지로 구성된 데이터 구조 화살표는 방향이 있는 엣지 
# 함수를 호출할 때 별표를 붙였는데, 이렇게 하면 리스트 언팩이 이루어짐. 언팩은 리스트의 원소를 낱개로 풀어서 전달하는 기법 
#--------------------------------------------------------------------------------------
# Function
#--------------------------------------------------------------------------------------     
# __call__ 파이썬의 특수 메서드 
class Function:
    def __call__(self, *inputs): # 인수에 별표를 붙이면 호출할 때 넘긴 인수들을 하나로 모아서 받을 수 있다. 
        inputs = [as_variable(x) for x in inputs] # 3/8
        xs = [x.data for x in inputs] # 데이터를 꺼낸다. 
                                      # 데이터를 여러개 처리할 수 있게 변경
        # y = x ** 2 # 실제 계산
        ys = self.forward(*xs) # 구체적인 계산은 forward 메서드에서 한다. 
                               # 별표를 붙여 언팩
        if not isinstance(ys, tuple): #튜플이 아닌 경우 추가 지원
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys] # Variable 형태로 되돌린다. 
        
        if Config.enable_backprop: # True 일때만 역전파코드 실행   
            self.generation = max([x.generation for x in inputs]) #세대 설정
            for output in outputs:
                output.set_creator(self) # 연결 설정 
            self.inputs = inputs #입력 변수를 기억(보관)한다.
            self.outputs = [weakref.ref(output) for output in outputs] # 출력도 저장한다.
                                                                       # output에 약한 참조방식을 적용(참조카운드가올라가지 않음)
        return outputs if len(outputs) > 1 else outputs[0] # 출력 또한 여러개 
    
    # 순전파
    # forward 메서드를 직접 호출한 사람에게 '이 메서드는 상속해서 구현해야 한다'는 사실을 알려준다. 
    def forward(self, xs):
        raise NotImplementedError() # 예외 발생 
    # 역전파 
    def backward(self, gys):
        raise NotImplementedError()


#합성합수 : 여러함수를 순서대로 적용하여 만들어진 변환 전체를 하나의 큰 함수로 볼 수 있다. 
#--------------------------------------------------------------------------------------
# 함수 클래스들
#--------------------------------------------------------------------------------------    
class Square(Function):
    def forward(self, x):
        y = x ** 2 
        return y        
    
    #y=x^2의 미분은 dy/dx=2x
    def backward(self, gy):
        x = self.inputs[0].data # 수정 전: x = self.input.data
        gx = 2 * x * gy
        return gx

# y=e^x (e는 자연로그 밑 2.718..오일러의 수, 네이피어 상수)
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    #y=e^x의 미분은 dy/dx = e^x
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)  
    
    def backward(self, gy):
        return gy, gy

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0
#--------------------------------------------------------------------------------------
# 여러가지 필요 함수 
#-------------------------------------------------------------------------------------- 
# 미분은 극한으로 짧은 시간(순간)에서의 변화량
# 수치미분 : 컴퓨터는 극한을 취급할 수 없으니 h를 극한과 비슷한 값으로 대체. 미세한 차이를 이용
# 중앙차분 : 수치미분에서 발생한 근사오차를 줄이는 방법. 전진차분(x-h)와 x+h에서의 기울기 구하는 방법
# 중앙차분 = f(x+h)-f(x-h) / 2h 
# 수치미분의 단점 : 자리수누락, 계산량이 많다. 그래서 등장한 역전파. 역전파로 구현하더라도 결과는 수치미분으로 확인 
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

#합성 함수의 미분
def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

def add(x0, x1):
    x1 = as_array(x1) # float, int, np.float64, np.int64와 같은 타입을 ndarray로 변경 
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1) # float, int, np.float64, np.int64와 같은 타입을 ndarray로 변경 
    return Mul()(x0, x1)

# x는 0차원 ndarray이지만, 제곱을 하면 np.float64가 되버린다. 
# 항상 ndarray인스턴스라고 가정했기 때문에 output에 사용했다. 
def as_array(x):
    if np.isscalar(x): # 입력데이터가 스칼라인지 판단 
        return np.array(x)
    return x

def as_variable(obj): #ndarray int float를 Variable인스턴스로 변환
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def rsub(x0, x1):
    x1 = as_array(x1)
 
# 연산자를 그대로 사용할 수 있게 추가 
Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__rsub__ = rsub    