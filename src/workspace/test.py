import jax

import jax.numpy as jnp

def fun(x,y):
    return x**3 + y**2

grad0 =jax.grad(fun, argnums=0)

print(jax.grad(fun)(3.0,4.0))  # Should print 43

grad1 =jax.grad(fun, argnums=1)

print(jax.grad(fun)(3.0,4.0))  # Should print 35

# jax.hessian(fun, argnums=0)

# print(jax.hessian(fun)(3.0))  # Should print 18.0