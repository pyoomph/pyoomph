Custom functions with multiple return values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The aforementioned :py:class:`~pyoomph.expressions.cb.CustomMathExpression` can only return a single scalar value at each time. Sometimes, however, you want to obtain more than one return value. Of course, one could implement one :py:class:`~pyoomph.expressions.cb.CustomMathExpression` for each of these values, but a lot of computations might be the same for each of these values. In this case, it is beneficial to use another class to obtain multiple values at once, namely the ``CustomMultiReturnExpression``. This is not explained in detail here, but we refer to the code in :py:mod:`pyoomph.expressions.tensor_funcs`.
