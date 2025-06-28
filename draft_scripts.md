h1(B, D) w2(D, K) h2 = h1 @ w2 (B, K)  
h1 @ w2 = h2  
loss = h2 - hlabel  
dloss / dh2
dloss/dw2 = dloss/dh2 @ dh2/dw2 = h2.grad @ h1.T

$$
\mathbf{h_1(B, D)}\times\mathbf{W_2(D, K)} = \mathbf{h_2(B, K)}\\
$$

$$
\mathbf{Loss} = \mathbf{h_2} - \mathbf{h_{label}}\\
$$

$$
\frac{\partial{\mathbf{Loss}}}{\partial{\mathbf{W_2}}} = \frac{\partial{\mathbf{Loss}}}{\partial{\mathbf{h_2}}}\times{\frac{\partial{\mathbf{h2}}}{\partial{\mathbf{W_2}}}}
$$

$$
\partial{\mathbf{Loss}}
$$